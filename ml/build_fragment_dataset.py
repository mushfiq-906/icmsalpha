"""
build_fragment_dataset.py
------------------------------------------------------------------------
Reads raw per-revision GlobalClones files and constructs a *fragment-level*
forecasting dataset from scratch.

Prediction unit
---------------
  Each row = one clone fragment at the revision where it changed.

Label
-----
  dependent   (0) - c1 changed AND at least one clone-class peer also
                     changed at the same revision (co-change)
  independent (1) - c1 changed ALONE (no peer changed at that revision)

Design rules
------------
  1. Dead peers excluded - skip fragments whose all peers are dead at r_t.
  2. Features from [1, r_t-1] only - the change at r_t is the label.
  3. Warm-up: >= 1 prior change before a fragment is eligible.
     (First change of every fragment is skipped.)

Usage
-----
  python ml/build_fragment_dataset.py --system Ctags --type Type3 --gran Block
  python ml/build_fragment_dataset.py --system Jmol  --type Type3 --gran Block
"""

import argparse
import math
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

# ── CONFIG ───────────────────────────────────────────────────────────────────

WORK_FOLDER = Path(__file__).resolve().parent.parent / "WorkFolder"
MIN_PRIOR_CHANGES = 1          # warm-up: fragment must have >=1 prior change
HALF_LIVES = [10, 20, 50]     # short / medium / long decay
RECENT_WINDOW = 5              # last-N change events for recent_co_ratio

# ── DATA LOADING ─────────────────────────────────────────────────────────────

def load_global_clones(system: str, ctype: str, gran: str):
    """Parse all GlobalClones/{rev}_clones.txt files."""
    clone_dir = WORK_FOLDER / system / "Clones" / ctype / gran / "GlobalClones"
    if not clone_dir.exists():
        raise FileNotFoundError(f"Clone directory not found: {clone_dir}")

    revision_clones = {}
    all_revisions = []

    for f in sorted(clone_dir.glob("*_clones.txt")):
        rev = int(f.stem.replace("_clones", ""))
        all_revisions.append(rev)
        clones = []
        with open(f, "r", encoding="latin-1") as fh:
            fh.readline()  # skip header
            for line in fh:
                parts = line.strip().split("|")
                if len(parts) < 9:
                    continue
                try:
                    clones.append({
                        "revision":       int(parts[0].strip()),
                        "filePath":       parts[1].strip(),
                        "changeType":     parts[2].strip(),
                        "startLine":      int(parts[3].strip()),
                        "endLine":        int(parts[4].strip()),
                        "nlines":         int(parts[5].strip()),
                        "cloneId":        int(parts[6].strip()),
                        "classId":        int(parts[7].strip()),
                        "globalCloneId":  int(parts[8].strip()),
                        "similarity":     int(parts[9].strip()) if len(parts) > 9 else 100,
                    })
                except (ValueError, IndexError):
                    continue
        revision_clones[rev] = clones

    all_revisions.sort()
    return revision_clones, all_revisions


def load_global_clone_info(system: str, ctype: str, gran: str, revisions: list):
    """Parse GlobalCloneInfo for method/package/class info."""
    info_dir = WORK_FOLDER / system / "Clones" / ctype / gran / "GlobalCloneInfo"
    result = {}
    if not info_dir.exists():
        return result

    for rev in revisions:
        f = info_dir / f"{rev}_clones.txt"
        if not f.exists():
            continue
        with open(f, "r", encoding="latin-1") as fh:
            header_line = fh.readline()
            if not header_line:
                continue
            header_parts = [h.strip().lower() for h in header_line.split("|")]
            idx = {col: i for i, col in enumerate(header_parts)}

            for line in fh:
                parts = line.strip().split("|")
                if len(parts) < 5:
                    continue
                try:
                    gcid = int(parts[0].strip())
                except (ValueError, IndexError):
                    continue
                if gcid in result:
                    continue
                info = {}
                for key in ["method_name", "method_signature", "package_name", "class_name"]:
                    if key in idx and idx[key] < len(parts):
                        val = parts[idx[key]].strip()
                        info[key] = val if val != "NULL" else None
                    else:
                        info[key] = None
                result[gcid] = info
    return result


def load_commit_log(system: str):
    """Parse commit_logs.txt -> dict[rev] -> author."""
    p = WORK_FOLDER / system / "commit_logs.txt"
    if not p.exists():
        return {}
    rev_author = {}
    with open(p, "r", encoding="latin-1") as fh:
        for line in fh:
            parts = line.strip().split("|")
            if len(parts) < 6:
                continue
            try:
                rev_author[int(parts[0].strip())] = parts[5].strip()
            except (ValueError, IndexError):
                continue
    return rev_author


def load_change_info_churn(system: str, rev: int):
    """Parse ChangeInfo -> dict[filename] -> lines_changed."""
    p = WORK_FOLDER / system / "Revisions" / "ChangeRevisionInfo" / "ChangeInfo" / f"revision_{rev}.txt"
    churn = defaultdict(int)
    if not p.exists():
        return churn
    with open(p, "r", encoding="latin-1") as fh:
        for line in fh:
            parts = line.strip().split("|")
            if len(parts) < 8:
                continue
            curr_file = parts[6].strip()
            curr_range = parts[7].strip()
            bare = curr_file.split("/")[-1].split("\\")[-1]
            lines = 1
            if "," in curr_range:
                try:
                    lo, hi = curr_range.split(",")
                    lines = abs(int(hi.strip()) - int(lo.strip())) + 1
                except ValueError:
                    pass
            churn[bare] += lines
    return churn


def extract_filename(filepath: str) -> str:
    if not filepath:
        return ""
    return filepath.replace("\\", "/").split("/")[-1]


def calculate_path_depth(path1: str, path2: str) -> int:
    """Folder distance between two clone file paths."""
    import re
    if not path1 or not path2:
        return -1
    path1 = re.sub(r"Revision_\d+/", "", path1.replace("\\", "/"))
    path2 = re.sub(r"Revision_\d+/", "", path2.replace("\\", "/"))
    if path1 == path2:
        return 0
    dir1 = path1.split("/")[:-1]
    dir2 = path2.split("/")[:-1]
    common = 0
    for a, b in zip(dir1, dir2):
        if a == b:
            common += 1
        else:
            break
    return (len(dir1) - common) + (len(dir2) - common) or 1


# ── HISTORY BUILDING ─────────────────────────────────────────────────────────

class FragmentHistory:
    __slots__ = [
        "gcid", "classId", "addedInRev", "deletedInRev", "endRev",
        "changeRevs", "unchangedRevs", "filePath", "nlines", "similarity",
    ]

    def __init__(self, gcid, classId):
        self.gcid = gcid
        self.classId = classId
        self.addedInRev = 0
        self.deletedInRev = 0
        self.endRev = 0
        self.changeRevs = []
        self.unchangedRevs = []
        self.filePath = ""
        self.nlines = 0
        self.similarity = 100


def build_histories(revision_clones: dict, all_revisions: list):
    """Build FragmentHistory for every globalCloneId across all revisions."""
    histories = {}
    class_members = defaultdict(set)
    rev_status = {}
    active_gcids = set()

    for rev in all_revisions:
        clones = revision_clones.get(rev, [])
        current_gcids = set()
        status = {}

        for c in clones:
            gcid = c["globalCloneId"]
            classId = c["classId"]
            current_gcids.add(gcid)
            status[gcid] = c["changeType"]

            if gcid not in histories:
                h = FragmentHistory(gcid, classId)
                h.addedInRev = rev
                h.filePath = c["filePath"]
                h.nlines = c["nlines"]
                h.similarity = c["similarity"]
                histories[gcid] = h
            else:
                h = histories[gcid]

            h.endRev = rev
            class_members[classId].add(gcid)

            if c["changeType"] == "M":
                h.changeRevs.append(rev)
            elif c["changeType"] == "U":
                h.unchangedRevs.append(rev)

        for gcid in active_gcids - current_gcids:
            h = histories[gcid]
            if h.deletedInRev == 0:
                h.deletedInRev = rev

        active_gcids = current_gcids
        rev_status[rev] = status

    for gcid in active_gcids:
        histories[gcid].deletedInRev = 0

    return histories, class_members, rev_status


# ── FEATURE COMPUTATION ──────────────────────────────────────────────────────

def lambda_from_half_life(h: int) -> float:
    return math.log(2.0) / h


def compute_decay_weighted(change_revs_before: list, target_set: set,
                           current_rev: int, lam: float) -> float:
    """Decay-weighted ratio: what fraction of prior changes are in target_set?"""
    w_target = 0.0
    w_total = 0.0
    for r in change_revs_before:
        w = math.exp(-lam * (current_rev - r))
        w_total += w
        if r in target_set:
            w_target += w
    if w_total < 1e-9:
        return 0.0
    return w_target / w_total


def compute_features(
    gcid: int, r_t: int, history: FragmentHistory,
    class_members: dict, histories: dict, rev_status: dict,
    clone_info: dict, churn_cache: dict, rev_author: dict, system: str,
) -> dict:
    """Compute all features for fragment gcid at revision r_t.
    All features use ONLY history from [1, r_t - 1]."""

    classId = history.classId
    all_peers = class_members.get(classId, set()) - {gcid}

    # Prior changes: only those strictly before r_t
    prior_changes = [r for r in history.changeRevs if r < r_t]
    n_prior = len(prior_changes)

    # Co-change history: for each prior change, did ANY peer also change?
    co_change_revs = set()
    indep_revs = set()
    for r in prior_changes:
        rs = rev_status.get(r, {})
        if any(rs.get(p) == "M" for p in all_peers):
            co_change_revs.add(r)
        else:
            indep_revs.add(r)

    n_co = len(co_change_revs)
    n_indep = len(indep_revs)
    co_ratio = n_co / n_prior if n_prior > 0 else 0.0

    # Temporal features
    last_change_rev = max(prior_changes) if prior_changes else history.addedInRev
    revs_since_last_change = r_t - last_change_rev

    co_sorted = sorted(co_change_revs)
    last_co_rev = co_sorted[-1] if co_sorted else -1
    revs_since_last_co = (r_t - last_co_rev) if last_co_rev > 0 else -1

    indep_sorted = sorted(indep_revs)
    last_indep_rev = indep_sorted[-1] if indep_sorted else -1
    revs_since_last_indep = (r_t - last_indep_rev) if last_indep_rev > 0 else -1

    # Solo events since last co-change
    if last_co_rev > 0:
        solo_since_co = sum(1 for r in indep_revs if r > last_co_rev)
    else:
        solo_since_co = n_indep

    # Recent co-change ratio (last RECENT_WINDOW change events)
    recent_changes = prior_changes[-RECENT_WINDOW:]
    if recent_changes:
        recent_co_ratio = sum(1 for r in recent_changes if r in co_change_revs) / len(recent_changes)
    else:
        recent_co_ratio = 0.0

    # Fragment properties
    file_age = r_t - history.addedInRev
    change_proneness = n_prior / file_age if file_age > 0 else 0.0
    total_alive = history.endRev - history.addedInRev + 1 if history.endRev >= history.addedInRev else 1
    prior_unchanged = len([r for r in history.unchangedRevs if r < r_t])
    stability_index = prior_unchanged / total_alive if total_alive > 0 else 0.0

    # Alive peers at r_t
    alive_peers = [p for p in all_peers
                   if histories[p].addedInRev <= r_t
                   and (histories[p].deletedInRev == 0 or histories[p].deletedInRev > r_t)]
    alive_peer_count = len(alive_peers)
    class_size = len(class_members.get(classId, set()))

    # Peer-aggregate features
    frac_same_file = 0.0
    frac_same_method = 0.0
    peer_co_ratios = []
    peer_depths = []

    my_filename = extract_filename(history.filePath)
    my_method = clone_info.get(gcid, {}).get("method_name") if clone_info else None

    for peer in alive_peers:
        ph = histories[peer]
        peer_filename = extract_filename(ph.filePath)
        if my_filename and peer_filename and my_filename == peer_filename:
            frac_same_file += 1
        peer_method = clone_info.get(peer, {}).get("method_name") if clone_info else None
        if my_method and peer_method and my_method == peer_method:
            frac_same_method += 1

        # Per-pair co-change ratio
        pair_co = sum(1 for r in prior_changes if rev_status.get(r, {}).get(peer) == "M")
        peer_co_ratios.append(pair_co / n_prior if n_prior > 0 else 0.0)
        peer_depths.append(calculate_path_depth(history.filePath, ph.filePath))

    if alive_peer_count > 0:
        frac_same_file /= alive_peer_count
        frac_same_method /= alive_peer_count
    mean_peer_co = np.mean(peer_co_ratios) if peer_co_ratios else 0.0
    max_peer_co = max(peer_co_ratios) if peer_co_ratios else 0.0
    min_peer_co = min(peer_co_ratios) if peer_co_ratios else 0.0
    mean_depth = np.mean(peer_depths) if peer_depths else -1.0

    # Class co-change density
    class_co_density = 0.0
    if alive_peer_count > 0:
        n_active = 0
        for peer in alive_peers:
            peer_prior = [r for r in histories[peer].changeRevs if r < r_t]
            for r in peer_prior:
                rs = rev_status.get(r, {})
                others = class_members.get(classId, set()) - {peer}
                if any(rs.get(p) == "M" for p in others):
                    n_active += 1
                    break
        class_co_density = n_active / alive_peer_count

    # Decay-weighted features
    decay_features = {}
    for hl in HALF_LIVES:
        lam = lambda_from_half_life(hl)
        decay_features[f"ir_decay_h{hl}"] = compute_decay_weighted(prior_changes, indep_revs, r_t, lam)
        decay_features[f"cs_decay_h{hl}"] = compute_decay_weighted(prior_changes, co_change_revs, r_t, lam)

    # Coupling trend
    mid_idx = len(prior_changes) // 2
    if mid_idx >= 2:
        early = prior_changes[:mid_idx]
        recent = prior_changes[mid_idx:]
        early_co = sum(1 for r in early if r in co_change_revs) / len(early)
        recent_co = sum(1 for r in recent if r in co_change_revs) / len(recent)
        coupling_trend = recent_co - early_co
    else:
        coupling_trend = 0.0

    # Process metrics
    my_bare = extract_filename(history.filePath)
    total_churn = 0
    for r in prior_changes:
        if r not in churn_cache:
            churn_cache[r] = load_change_info_churn(system, r)
        total_churn += churn_cache[r].get(my_bare, 0)

    # Ownership
    author_counts = Counter()
    for r in prior_changes:
        author_counts[rev_author.get(r, "UNKNOWN")] += 1
    distinct_authors = len(author_counts)
    major_author_prop = max(author_counts.values()) / n_prior if n_prior > 0 and author_counts else 0.0

    return {
        "n_prior_changes":           n_prior,
        "n_prior_co_changes":        n_co,
        "n_prior_independent":       n_indep,
        "co_change_ratio":           round(co_ratio, 4),
        "revs_since_last_change":    revs_since_last_change,
        "revs_since_last_co_change": revs_since_last_co,
        "revs_since_last_independent": revs_since_last_indep,
        "recent_co_ratio_last5":     round(recent_co_ratio, 4),
        "solo_events_since_last_co": solo_since_co,
        "nlines":                    history.nlines,
        "similarity":                history.similarity,
        "file_age":                  file_age,
        "change_proneness":          round(change_proneness, 4),
        "stability_index":           round(stability_index, 4),
        "class_size":                class_size,
        "alive_peer_count":          alive_peer_count,
        "class_co_change_density":   round(class_co_density, 4),
        "frac_peers_same_file":      round(frac_same_file, 4),
        "frac_peers_same_method":    round(frac_same_method, 4),
        "mean_peer_co_ratio":        round(mean_peer_co, 4),
        "max_peer_co_ratio":         round(max_peer_co, 4),
        "min_peer_co_ratio":         round(min_peer_co, 4),
        "mean_peer_depth":           round(mean_depth, 2),
        **{k: round(v, 4) for k, v in decay_features.items()},
        "coupling_trend":            round(coupling_trend, 4),
        "churn":                     total_churn,
        "distinct_authors":          distinct_authors,
        "major_author_prop":         round(major_author_prop, 4),
    }


# ── DATASET CONSTRUCTION ─────────────────────────────────────────────────────

def build_dataset(system: str, ctype: str, gran: str):
    print(f"=== Building fragment dataset: {system} / {ctype} / {gran} ===")

    print("Loading GlobalClones...")
    revision_clones, all_revisions = load_global_clones(system, ctype, gran)
    print(f"  {len(all_revisions)} revisions, "
          f"{sum(len(v) for v in revision_clones.values()):,} total clone-revision records")

    print("Building histories...")
    histories, class_members, rev_status = build_histories(revision_clones, all_revisions)
    print(f"  {len(histories)} unique fragments, {len(class_members)} clone classes")

    print("Loading GlobalCloneInfo (method data)...")
    clone_info = load_global_clone_info(system, ctype, gran, all_revisions)
    print(f"  Method info for {len(clone_info)} fragments")

    print("Loading commit log...")
    rev_author = load_commit_log(system)
    print(f"  Author info for {len(rev_author)} revisions")

    churn_cache = {}

    # Phase 2: Build prediction instances
    # One row per changed fragment.
    # Label: dependent (0) if ANY alive peer also changed at r_t
    #        independent (1) if the fragment changed ALONE
    print("\nBuilding prediction instances...")
    rows = []
    stats = {"total_changes": 0, "dead_peers": 0, "warmup_skip": 0, "emitted": 0}

    changed_revisions = [rev for rev in all_revisions
                         if any(ct == "M" for ct in rev_status.get(rev, {}).values())]
    print(f"  {len(changed_revisions)} changed revisions (out of {len(all_revisions)} total)")

    for rev_idx, r_t in enumerate(changed_revisions):
        rs = rev_status.get(r_t, {})
        changed_gcids = [gcid for gcid, ct in rs.items() if ct == "M"]

        for gcid in changed_gcids:
            stats["total_changes"] += 1
            h = histories.get(gcid)
            if h is None:
                continue

            classId = h.classId
            all_peers = class_members.get(classId, set()) - {gcid}

            # Alive peers at r_t
            alive_peers = [p for p in all_peers
                           if histories[p].addedInRev <= r_t
                           and (histories[p].deletedInRev == 0 or histories[p].deletedInRev > r_t)]
            if len(alive_peers) == 0:
                stats["dead_peers"] += 1
                continue

            # Warm-up: need >= MIN_PRIOR_CHANGES prior changes
            prior_changes = [r for r in h.changeRevs if r < r_t]
            if len(prior_changes) < MIN_PRIOR_CHANGES:
                stats["warmup_skip"] += 1
                continue

            # LABEL: did ANY alive peer also change (M) at r_t?
            any_peer_changed = any(rs.get(p) == "M" for p in alive_peers)
            label = 0 if any_peer_changed else 1  # 0=dependent, 1=independent

            # Features from [1, r_t - 1] only
            features = compute_features(
                gcid, r_t, h, class_members, histories,
                rev_status, clone_info, churn_cache, rev_author, system,
            )

            rows.append({
                "Revision": r_t,
                "gcid": gcid,
                "classid": classId,
                **features,
                "label": label,
            })
            stats["emitted"] += 1

        if (rev_idx + 1) % 50 == 0:
            print(f"  Processed {rev_idx + 1}/{len(changed_revisions)} changed revisions "
                  f"({stats['emitted']} instances so far)")

    # Save
    df = pd.DataFrame(rows)
    df = df.sort_values(["Revision", "gcid"]).reset_index(drop=True)

    print(f"\n{'='*60}")
    print(f"  DATASET SUMMARY: {system} / {ctype} / {gran}")
    print(f"{'='*60}")
    print(f"  Total fragment change events : {stats['total_changes']:,}")
    print(f"  Excluded (all peers dead)    : {stats['dead_peers']:,}")
    print(f"  Excluded (warm-up < {MIN_PRIOR_CHANGES})      : {stats['warmup_skip']:,}")
    print(f"  Final prediction instances   : {stats['emitted']:,}")
    print(f"  Unique fragments             : {df['gcid'].nunique()}")
    print(f"  Unique revisions             : {df['Revision'].nunique()}")
    dep = (df["label"] == 0).sum()
    ind = (df["label"] == 1).sum()
    print(f"  Label balance                : {dep} dependent / {ind} independent "
          f"({ind/(dep+ind):.1%} independent)")
    print(f"  Revision range               : {df['Revision'].min()} - {df['Revision'].max()}")
    print(f"  Columns                      : {df.shape[1]}")

    out_dir = WORK_FOLDER / system / "Datasets" / "CloneGenealogy"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ctype}_{gran}_fragment_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Saved -> {out_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build fragment-level forecasting dataset from raw GlobalClones data")
    parser.add_argument("--system", default="Ctags",
                        help="Subject system folder name (e.g., Ctags, Jmol)")
    parser.add_argument("--type", default="Type3",
                        help="Clone type (Type1, Type2, Type3)")
    parser.add_argument("--gran", default="Block",
                        help="Granularity (Block, Function)")
    args = parser.parse_args()
    build_dataset(args.system, args.type, args.gran)
