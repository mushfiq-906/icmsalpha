"""
build_classifier_dataset.py
------------------------------------------------------------------------
Builds a fresh per-(gcid, decision-event) dataset for the new classifier
methodology: classify a single clone fragment as dependent (tends to
co-change with class peers) or independent (tends to change alone), then
verify the prediction at the fragment's NEXT actual change revision.

Each row = one decision event = one revision M where the fragment changed
(real "M" — see below) AFTER the warmup of MIN_PRIOR_CHANGES prior changes.

NiCad's per-file changeType="M" flags every clone in a touched file even
when its source text is identical (a line-shift artifact). To avoid that,
we re-verify every "M" by reading the actual fragment text at revision M
and at the previous revision where the gcid was seen, and only keep "M"
if the texts truly differ. This is "Option C" in the methodology design.

Features use ONLY history strictly before M (revisions < M).

Label convention:
    label = 1  ->  dependent    (>=1 alive class peer has REAL "M" at M)
    label = 0  ->  independent  (no peer had a real change at M)

Companion output: {Type}_{Gran}_classifier_excluded.csv lists fragments
that never reached the warmup or were always skipped.

Usage
-----
  python ml/build_classifier_dataset.py --system Ctags --type Type3 --gran Block
  python ml/build_classifier_dataset.py --system Jmol  --type Type3 --gran Block \\
      --source-root "D:/Thesis/NiCad-6.2/systems/Jmol"
"""

import argparse
import math
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

WORK_FOLDER = Path(__file__).resolve().parent.parent / "WorkFolder"
DEFAULT_SOURCE_ROOT_TEMPLATE = "D:/Thesis/NiCad-6.2/systems/{system}"
MIN_PRIOR_CHANGES = 1
HALF_LIVES = [10, 20, 50]
RECENT_WINDOW = 5
# Revisions where more than this many fragments REALLY changed are dropped:
# project-wide refactors (license headers, API renames) trivially make every
# fragment look "dependent" even though there is no clone-coupling signal.
BULK_EDIT_THRESHOLD = 30


# ── DATA LOADING ─────────────────────────────────────────────────────────────

def load_global_clones(system: str, ctype: str, gran: str):
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


_FILE_LINES_CACHE = {}


def read_file_lines(source_root: Path, file_path: str):
    """Read all lines of a file at <source_root>/<file_path>. Cached.
    Returns list[str] of lines (with trailing newlines stripped). Returns
    None if the file is missing."""
    abs_path = (source_root / file_path).resolve()
    cached = _FILE_LINES_CACHE.get(abs_path)
    if cached is not None:
        return cached
    if not abs_path.exists():
        _FILE_LINES_CACHE[abs_path] = None
        return None
    try:
        with open(abs_path, "r", encoding="latin-1") as fh:
            lines = [ln.rstrip("\r\n") for ln in fh.readlines()]
    except OSError:
        lines = None
    _FILE_LINES_CACHE[abs_path] = lines
    return lines


def get_fragment_text(source_root: Path, file_path: str,
                      start_line: int, end_line: int) -> str:
    """Return whitespace-normalized text of a fragment, or '' if unreadable."""
    lines = read_file_lines(source_root, file_path)
    if lines is None:
        return ""
    s = max(1, start_line) - 1
    e = min(len(lines), end_line)
    if e < s:
        return ""
    snippet = lines[s:e]
    # Strip trailing whitespace per line and drop blank lines so that
    # whitespace-only edits / line-ending differences don't count as changes.
    norm = [ln.rstrip() for ln in snippet]
    norm = [ln for ln in norm if ln.strip()]
    return "\n".join(norm)


def verify_real_changes(revision_clones: dict, all_revisions: list,
                        source_root: Path):
    """For every (rev, gcid) marked changeType=='M', compare the fragment's
    current text to its text at the previous revision where it appeared.
    Returns: dict[rev][gcid] -> bool (True if the M is a real modification).

    Also returns counters for reporting."""
    # Per-gcid timeline of (rev, filePath, startLine, endLine, changeType)
    gcid_timeline = defaultdict(list)
    for rev in all_revisions:
        for c in revision_clones.get(rev, []):
            gcid_timeline[c["globalCloneId"]].append(
                (rev, c["filePath"], c["startLine"], c["endLine"], c["changeType"]))

    real_m = defaultdict(dict)
    stats = {"m_total": 0, "m_real": 0, "m_spurious": 0, "m_unverifiable": 0}

    for gcid, timeline in gcid_timeline.items():
        for i, (rev, fp, s, e, ct) in enumerate(timeline):
            if ct != "M":
                continue
            stats["m_total"] += 1
            if i == 0:
                # First-ever appearance flagged M (rare). Treat as real.
                real_m[rev][gcid] = True
                stats["m_real"] += 1
                continue
            prev_rev, prev_fp, prev_s, prev_e, _ = timeline[i - 1]
            curr_text = get_fragment_text(source_root, fp, s, e)
            prev_text = get_fragment_text(source_root, prev_fp, prev_s, prev_e)
            if not curr_text or not prev_text:
                # Couldn't read either side — be conservative, treat as real.
                real_m[rev][gcid] = True
                stats["m_unverifiable"] += 1
                continue
            if curr_text == prev_text:
                real_m[rev][gcid] = False
                stats["m_spurious"] += 1
            else:
                real_m[rev][gcid] = True
                stats["m_real"] += 1

    return real_m, stats


def calculate_path_depth(path1: str, path2: str) -> int:
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


def build_histories(revision_clones: dict, all_revisions: list, real_m: dict):
    """Build histories using real_m to override per-file 'M' flags. A clone
    record marked 'M' is downgraded to 'U' if its fragment text didn't actually
    change vs the previous revision where the gcid was seen."""
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

            raw_ct = c["changeType"]
            # Override: if "M" but text didn't really change, treat as "U"
            if raw_ct == "M" and not real_m.get(rev, {}).get(gcid, True):
                eff_ct = "U"
            else:
                eff_ct = raw_ct
            status[gcid] = eff_ct

            if gcid not in histories:
                h = FragmentHistory(gcid, classId)
                h.addedInRev = rev
                h.filePath = c["filePath"]
                h.nlines = c["nlines"]
                h.similarity = c["similarity"]
                histories[gcid] = h
            else:
                h = histories[gcid]
                h.filePath = c["filePath"]
                h.nlines = c["nlines"]
                h.similarity = c["similarity"]

            h.endRev = rev
            class_members[classId].add(gcid)

            if eff_ct == "M":
                h.changeRevs.append(rev)
            elif eff_ct == "U":
                h.unchangedRevs.append(rev)

        for gcid in active_gcids - current_gcids:
            h = histories[gcid]
            if h.deletedInRev == 0:
                h.deletedInRev = rev

        active_gcids = current_gcids
        rev_status[rev] = status

    return histories, class_members, rev_status


# ── FEATURE COMPUTATION ──────────────────────────────────────────────────────

def lambda_from_half_life(h: int) -> float:
    return math.log(2.0) / h


def compute_decay_weighted(change_revs_before: list, target_set: set,
                           current_rev: int, lam: float) -> float:
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
    """Compute all features for fragment gcid at decision-event revision r_t.
    All features use ONLY history from revisions strictly < r_t."""

    classId = history.classId
    all_peers = class_members.get(classId, set()) - {gcid}

    prior_changes = [r for r in history.changeRevs if r < r_t]
    n_prior = len(prior_changes)

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
    solo_ratio = n_indep / n_prior if n_prior > 0 else 0.0

    last_change_rev = max(prior_changes) if prior_changes else history.addedInRev
    revs_since_last_change = r_t - last_change_rev

    co_sorted = sorted(co_change_revs)
    last_co_rev = co_sorted[-1] if co_sorted else -1
    revs_since_last_co = (r_t - last_co_rev) if last_co_rev > 0 else -1

    indep_sorted = sorted(indep_revs)
    last_indep_rev = indep_sorted[-1] if indep_sorted else -1
    revs_since_last_indep = (r_t - last_indep_rev) if last_indep_rev > 0 else -1

    if last_co_rev > 0:
        solo_since_co = sum(1 for r in indep_revs if r > last_co_rev)
    else:
        solo_since_co = n_indep

    recent_changes = prior_changes[-RECENT_WINDOW:]
    if recent_changes:
        recent_co_ratio = sum(1 for r in recent_changes if r in co_change_revs) / len(recent_changes)
        recent_solo_ratio = 1.0 - recent_co_ratio
    else:
        recent_co_ratio = 0.0
        recent_solo_ratio = 0.0

    file_age = r_t - history.addedInRev
    change_density = n_prior / file_age if file_age > 0 else 0.0
    total_alive = history.endRev - history.addedInRev + 1 if history.endRev >= history.addedInRev else 1
    prior_unchanged = len([r for r in history.unchangedRevs if r < r_t])
    stability_index = prior_unchanged / total_alive if total_alive > 0 else 0.0

    alive_peers = [p for p in all_peers
                   if histories[p].addedInRev <= r_t
                   and (histories[p].deletedInRev == 0 or histories[p].deletedInRev > r_t)]
    alive_peer_count = len(alive_peers)
    class_size = len(class_members.get(classId, set()))

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

        pair_co = sum(1 for r in prior_changes if rev_status.get(r, {}).get(peer) == "M")
        peer_co_ratios.append(pair_co / n_prior if n_prior > 0 else 0.0)
        peer_depths.append(calculate_path_depth(history.filePath, ph.filePath))

    if alive_peer_count > 0:
        frac_same_file /= alive_peer_count
        frac_same_method /= alive_peer_count
    mean_peer_co = float(np.mean(peer_co_ratios)) if peer_co_ratios else 0.0
    max_peer_co = max(peer_co_ratios) if peer_co_ratios else 0.0
    min_peer_co = min(peer_co_ratios) if peer_co_ratios else 0.0
    mean_depth = float(np.mean(peer_depths)) if peer_depths else -1.0

    decay_features = {}
    for hl in HALF_LIVES:
        lam = lambda_from_half_life(hl)
        decay_features[f"independent_ratio_decay_h{hl}"] = compute_decay_weighted(
            prior_changes, indep_revs, r_t, lam)
        decay_features[f"coupling_strength_decay_h{hl}"] = compute_decay_weighted(
            prior_changes, co_change_revs, r_t, lam)

    mid_idx = len(prior_changes) // 2
    if mid_idx >= 2:
        early = prior_changes[:mid_idx]
        recent = prior_changes[mid_idx:]
        early_co = sum(1 for r in early if r in co_change_revs) / len(early)
        recent_co = sum(1 for r in recent if r in co_change_revs) / len(recent)
        coupling_trend = recent_co - early_co
    else:
        coupling_trend = 0.0

    my_bare = extract_filename(history.filePath)
    total_churn = 0
    for r in prior_changes:
        if r not in churn_cache:
            churn_cache[r] = load_change_info_churn(system, r)
        total_churn += churn_cache[r].get(my_bare, 0)

    author_counts = Counter()
    for r in prior_changes:
        author_counts[rev_author.get(r, "UNKNOWN")] += 1
    distinct_authors = len(author_counts)
    major_author_prop = max(author_counts.values()) / n_prior if n_prior > 0 and author_counts else 0.0

    return {
        "n_prior_changes":             n_prior,
        "n_prior_co_changes":          n_co,
        "n_prior_solo_changes":        n_indep,
        "co_change_ratio":             round(co_ratio, 4),
        "solo_change_ratio":           round(solo_ratio, 4),
        "revs_since_last_change":      revs_since_last_change,
        "revs_since_last_co_change":   revs_since_last_co,
        "revs_since_last_solo_change": revs_since_last_indep,
        "solo_events_since_last_co_change": solo_since_co,
        "recent_co_ratio_last5":       round(recent_co_ratio, 4),
        "recent_solo_ratio_last5":     round(recent_solo_ratio, 4),
        "nlines":                      history.nlines,
        "similarity":                  history.similarity,
        "file_age":                    file_age,
        "change_density":              round(change_density, 4),
        "stability_index":             round(stability_index, 4),
        "class_size":                  class_size,
        "alive_peer_count":            alive_peer_count,
        "frac_peers_same_file":        round(frac_same_file, 4),
        "frac_peers_same_method":      round(frac_same_method, 4),
        "mean_peer_co_ratio":          round(mean_peer_co, 4),
        "max_peer_co_ratio":           round(max_peer_co, 4),
        "min_peer_co_ratio":           round(min_peer_co, 4),
        "mean_peer_depth":             round(mean_depth, 2),
        **{k: round(v, 4) for k, v in decay_features.items()},
        "coupling_trend":              round(coupling_trend, 4),
        "cumulative_churn":            total_churn,
        "distinct_authors":            distinct_authors,
        "major_author_prop":           round(major_author_prop, 4),
    }


# ── DATASET CONSTRUCTION ─────────────────────────────────────────────────────

def build_dataset(system: str, ctype: str, gran: str, min_prior_changes: int,
                  source_root: str = None, bulk_edit_threshold: int = BULK_EDIT_THRESHOLD):
    print(f"=== Building classifier dataset: {system} / {ctype} / {gran} ===")
    print(f"    MIN_PRIOR_CHANGES = {min_prior_changes}")

    if source_root is None:
        source_root = DEFAULT_SOURCE_ROOT_TEMPLATE.format(system=system)
    source_root_path = Path(source_root)
    print(f"    SOURCE_ROOT       = {source_root_path}")
    if not source_root_path.exists():
        raise FileNotFoundError(
            f"Source root not found: {source_root_path}. Pass --source-root.")

    print("Loading GlobalClones...")
    revision_clones, all_revisions = load_global_clones(system, ctype, gran)
    print(f"  {len(all_revisions)} revisions, "
          f"{sum(len(v) for v in revision_clones.values()):,} total clone-revision records")

    print("Verifying real changes (Option C: text-level diff)...")
    real_m, real_stats = verify_real_changes(revision_clones, all_revisions, source_root_path)
    m_total = real_stats["m_total"]
    if m_total > 0:
        print(f"  M events total       : {m_total:,}")
        print(f"  Real M               : {real_stats['m_real']:,} "
              f"({real_stats['m_real']/m_total:.1%})")
        print(f"  Spurious M (line-shift): {real_stats['m_spurious']:,} "
              f"({real_stats['m_spurious']/m_total:.1%})")
        print(f"  Unverifiable (file missing): {real_stats['m_unverifiable']:,}")

    print("Building histories...")
    histories, class_members, rev_status = build_histories(
        revision_clones, all_revisions, real_m)
    print(f"  {len(histories)} unique fragments, {len(class_members)} clone classes")

    print("Loading GlobalCloneInfo (method data)...")
    clone_info = load_global_clone_info(system, ctype, gran, all_revisions)
    print(f"  Method info for {len(clone_info)} fragments")

    print("Loading commit log...")
    rev_author = load_commit_log(system)
    print(f"  Author info for {len(rev_author)} revisions")

    churn_cache = {}

    print("\nBuilding decision-event rows...")
    rows = []
    stats = {
        "total_changes": 0,
        "dead_peers": 0,
        "warmup_skip": 0,
        "emitted": 0,
    }

    eligible_gcids = set()
    skipped_dead_peers_gcids = set()

    for gcid, h in histories.items():
        if not h.changeRevs:
            continue
        all_peers = class_members.get(h.classId, set()) - {gcid}

        for i, r_t in enumerate(h.changeRevs):
            stats["total_changes"] += 1
            if i < min_prior_changes:
                stats["warmup_skip"] += 1
                continue

            alive_peers = [p for p in all_peers
                           if histories[p].addedInRev <= r_t
                           and (histories[p].deletedInRev == 0 or histories[p].deletedInRev > r_t)]
            if len(alive_peers) == 0:
                stats["dead_peers"] += 1
                skipped_dead_peers_gcids.add(gcid)
                continue

            rs = rev_status.get(r_t, {})
            any_peer_changed = any(rs.get(p) == "M" for p in alive_peers)
            label = 1 if any_peer_changed else 0  # 1=dependent, 0=independent

            features = compute_features(
                gcid, r_t, h, class_members, histories,
                rev_status, clone_info, churn_cache, rev_author, system,
            )

            prev_change_rev = h.changeRevs[i - 1]
            rows.append({
                "verification_revision": r_t,
                "gcid": gcid,
                "classid": h.classId,
                "decision_index": i + 1,  # 1-based: 2 means "this is the 2nd change"
                "prev_change_revision": prev_change_rev,
                "gap_since_prev_change": r_t - prev_change_rev,
                **features,
                "label": label,
            })
            stats["emitted"] += 1
            eligible_gcids.add(gcid)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["verification_revision", "gcid"]).reset_index(drop=True)

    # ── Bulk-edit filter ───────────────────────────────────────────────────
    # Drop any verification_revision where >threshold fragments really changed
    # codebase-wide.  Those are project-wide events (license headers, refactors)
    # where "any peer changed" fires spuriously for every fragment, flooding the
    # dataset with trivially-true dependent labels that the model cannot learn from.
    if bulk_edit_threshold > 0 and not df.empty:
        rev_real_m_count = {
            rev: sum(1 for ct in status.values() if ct == "M")
            for rev, status in rev_status.items()
        }
        bulk_edit_revs = {
            rev for rev, cnt in rev_real_m_count.items() if cnt > bulk_edit_threshold
        }
        rows_before = len(df)
        df = df[~df["verification_revision"].isin(bulk_edit_revs)].reset_index(drop=True)
        rows_dropped = rows_before - len(df)
        print(f"\n  Bulk-edit filter (threshold={bulk_edit_threshold}):")
        print(f"    Revisions with >={bulk_edit_threshold}+1 real M events : {len(bulk_edit_revs)}")
        print(f"    Decision rows dropped                     : {rows_dropped}")
        print(f"    Decision rows remaining                   : {len(df)}")

    # ── Companion: excluded fragments ──────────────────────────────────────
    excluded_rows = []
    for gcid, h in histories.items():
        if gcid in eligible_gcids:
            continue
        total_changes = len(h.changeRevs)
        end_rev = h.endRev
        all_peers = class_members.get(h.classId, set()) - {gcid}
        alive_at_end = sum(1 for p in all_peers
                           if histories[p].addedInRev <= end_rev
                           and (histories[p].deletedInRev == 0 or histories[p].deletedInRev > end_rev))
        if total_changes == 0:
            reason = "never_changed"
        elif total_changes <= min_prior_changes:
            reason = "warmup"
        elif gcid in skipped_dead_peers_gcids:
            reason = "no_alive_peers"
        else:
            reason = "warmup"
        lifespan = end_rev - h.addedInRev + 1 if end_rev >= h.addedInRev else 0
        excluded_rows.append({
            "gcid": gcid,
            "classid": h.classId,
            "addedInRev": h.addedInRev,
            "endRev": end_rev,
            "deletedInRev": h.deletedInRev,
            "lifespan": lifespan,
            "total_changes": total_changes,
            "alive_peer_count_at_end": alive_at_end,
            "exclusion_reason": reason,
        })
    excluded_df = pd.DataFrame(excluded_rows)

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  DATASET SUMMARY: {system} / {ctype} / {gran}")
    print(f"{'='*60}")
    print(f"  Total fragment change events : {stats['total_changes']:,}")
    print(f"  Excluded (warmup < {min_prior_changes}+1)     : {stats['warmup_skip']:,}")
    print(f"  Excluded (no alive peers)    : {stats['dead_peers']:,}")
    print(f"  Final decision events        : {stats['emitted']:,}")
    if not df.empty:
        print(f"  Unique gcids                 : {df['gcid'].nunique()}")
        print(f"  Unique verification revs     : {df['verification_revision'].nunique()}")
        dep = int((df["label"] == 1).sum())
        ind = int((df["label"] == 0).sum())
        total = dep + ind
        print(f"  Label balance                : {dep} dependent / {ind} independent "
              f"({dep/total:.1%} dependent)")
        print(f"  verification_revision range  : "
              f"{df['verification_revision'].min()} - {df['verification_revision'].max()}")
        di_counts = df["decision_index"].value_counts().sort_index()
        print(f"  decision_index distribution  : "
              f"{', '.join(f'{k}:{v}' for k, v in di_counts.head(8).items())}"
              f"{' ...' if len(di_counts) > 8 else ''}")
        print(f"  Columns                      : {df.shape[1]}")
    print(f"  Excluded fragments (companion): {len(excluded_df):,}")
    if not excluded_df.empty:
        for reason, n in excluded_df["exclusion_reason"].value_counts().items():
            print(f"    - {reason}: {n}")

    out_dir = WORK_FOLDER / system / "Datasets" / "CloneGenealogy"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ctype}_{gran}_classifier_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Saved main      -> {out_path}")

    excl_path = out_dir / f"{ctype}_{gran}_classifier_excluded.csv"
    excluded_df.to_csv(excl_path, index=False)
    print(f"  Saved excluded  -> {excl_path}")
    return df, excluded_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build per-fragment classifier dataset (dependent vs independent) "
                    "with companion exclusions list, from raw GlobalClones data.")
    parser.add_argument("--system", default="Ctags",
                        help="Subject system folder name (e.g., Ctags, Jmol)")
    parser.add_argument("--type", default="Type3",
                        help="Clone type (Type1, Type2, Type3)")
    parser.add_argument("--gran", default="Block",
                        help="Granularity (Block, Function)")
    parser.add_argument("--min-prior-changes", type=int, default=MIN_PRIOR_CHANGES,
                        help="Warmup: number of prior changes required before a fragment "
                             "becomes eligible for prediction. Default=1.")
    parser.add_argument("--source-root", default=None,
                        help="Root directory holding Revision_<n>/ subdirs with the actual "
                             "source files. Default: D:/Thesis/NiCad-6.2/systems/{system}.")
    parser.add_argument("--bulk-edit-threshold", type=int, default=BULK_EDIT_THRESHOLD,
                        help="Drop verification revisions where more than this many fragments "
                             "really changed (project-wide refactors). 0 = disabled. Default=30.")
    args = parser.parse_args()
    build_dataset(args.system, args.type, args.gran, args.min_prior_changes,
                  args.source_root, args.bulk_edit_threshold)
