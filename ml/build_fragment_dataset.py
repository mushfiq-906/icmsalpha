"""
build_fragment_dataset.py
------------------------------------------------------------------------
Reads raw per-revision GlobalClones files and constructs a fragment-level
forecasting dataset.

Prediction unit
---------------
  Each row = one clone fragment at the revision where it changed.

Label
-----
  dependent   (0) - c1 changed AND at least one clone-class peer also
                     really changed at the same revision (co-change)
  independent (1) - c1 changed ALONE (no peer had a real change at that revision)

M verification  (broader than NiCad's file-level flag)
---------------------------------------------------------------------------
  NiCad marks ALL clones in a touched file as "M" even when the fragment's
  own source text is identical (line-shift artifact).  We override with
  three-level verification before any label or feature is computed:

    Level 1 — whitespace normalisation
      Strip trailing whitespace per line and blank lines; if the normalized
      texts match the fragment did not change.

    Level 2 — comment stripping
      If Level 1 still differs, strip Java // single-line and /* */ block
      comments and compare again. If the stripped texts match, the change
      was comments-only and is treated as non-substantive (spurious).

    Level 3 — change magnitude
      When a change is real (Levels 1 & 2 both differ), measure what
      fraction of the fragment's lines actually changed (via difflib).
      This becomes the feature `change_magnitude` and also contributes
      to `change_volume` (raw count of changed lines).

  Bulk-edit filter
      After the dataset is built, revisions where more than BULK_EDIT_THRESHOLD
      fragments really changed are dropped.  Project-wide refactors (license
      headers, API renames) make the "any peer changed" label trivially true
      and are uninformative for clone-coupling analysis.

Design rules
------------
  1. Dead peers excluded — skip fragments whose ALL peers are dead at r_t.
  2. Features from [1, r_t-1] only — the change at r_t is the label.
  3. Warm-up: >= 1 prior change before a fragment is eligible.

Usage
-----
  python ml/build_fragment_dataset.py --system Ctags --type Type3 --gran Block
  python ml/build_fragment_dataset.py --system Jmol  --type Type3 --gran Block \\
      --source-root "D:/Thesis/NiCad-6.2/systems/Jmol"
"""

import argparse
import difflib
import math
import re
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

# ── CONFIG ───────────────────────────────────────────────────────────────────

WORK_FOLDER = Path(__file__).resolve().parent.parent / "WorkFolder"
DEFAULT_SOURCE_ROOT_TEMPLATE = "D:/Thesis/NiCad-6.2/systems/{system}"
MIN_PRIOR_CHANGES = 1
HALF_LIVES = [10, 20, 50]
RECENT_WINDOW = 5
BULK_EDIT_THRESHOLD = 30     # drop revisions with >N real changes codebase-wide

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
            fh.readline()
            for line in fh:
                parts = line.strip().split("|")
                if len(parts) < 9:
                    continue
                try:
                    clones.append({
                        "revision":      int(parts[0].strip()),
                        "filePath":      parts[1].strip(),
                        "changeType":    parts[2].strip(),
                        "startLine":     int(parts[3].strip()),
                        "endLine":       int(parts[4].strip()),
                        "nlines":        int(parts[5].strip()),
                        "cloneId":       int(parts[6].strip()),
                        "classId":       int(parts[7].strip()),
                        "globalCloneId": int(parts[8].strip()),
                        "similarity":    int(parts[9].strip()) if len(parts) > 9 else 100,
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
    p = (WORK_FOLDER / system / "Revisions" / "ChangeRevisionInfo"
         / "ChangeInfo" / f"revision_{rev}.txt")
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
    if not path1 or not path2:
        return -1
    path1 = re.sub(r"Revision_\d+/", "", path1.replace("\\", "/"))
    path2 = re.sub(r"Revision_\d+/", "", path2.replace("\\", "/"))
    if path1 == path2:
        return 0
    dir1 = path1.split("/")[:-1]
    dir2 = path2.split("/")[:-1]
    common = sum(1 for a, b in zip(dir1, dir2) if a == b)
    return (len(dir1) - common) + (len(dir2) - common) or 1


# ── SOURCE-TEXT UTILITIES ─────────────────────────────────────────────────────

_FILE_LINES_CACHE: dict = {}

_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT_RE  = re.compile(r"//[^\n]*")


def read_file_lines(source_root: Path, file_path: str):
    abs_path = (source_root / file_path).resolve()
    if abs_path in _FILE_LINES_CACHE:
        return _FILE_LINES_CACHE[abs_path]
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


def _extract_lines(source_root: Path, file_path: str,
                   start_line: int, end_line: int) -> list:
    """Return raw lines of a fragment (empty list if unreadable)."""
    lines = read_file_lines(source_root, file_path)
    if lines is None:
        return []
    s = max(1, start_line) - 1
    e = min(len(lines), end_line)
    return lines[s:e] if e >= s else []


def _normalize_ws(lines: list) -> str:
    """Level-1 normalization: strip trailing whitespace, drop blank lines."""
    normed = [ln.rstrip() for ln in lines]
    normed = [ln for ln in normed if ln.strip()]
    return "\n".join(normed)


def _strip_comments(text: str) -> str:
    """Level-2 normalization: remove Java block and line comments."""
    text = _BLOCK_COMMENT_RE.sub("", text)
    text = _LINE_COMMENT_RE.sub("", text)
    # Re-normalize whitespace after removal
    lines = [ln.rstrip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln.strip()]
    return "\n".join(lines)


def _change_magnitude(prev_lines: list, curr_lines: list) -> tuple:
    """Return (changed_lines, magnitude) using unified diff line counts.
    magnitude = fraction of max(len(prev), len(curr)) lines that changed."""
    if not prev_lines and not curr_lines:
        return 0, 0.0
    matcher = difflib.SequenceMatcher(None,
                                      [ln.rstrip() for ln in prev_lines],
                                      [ln.rstrip() for ln in curr_lines])
    changed = sum(max(i2 - i1, j2 - j1)
                  for tag, i1, i2, j1, j2 in matcher.get_opcodes()
                  if tag != "equal")
    denom = max(len(prev_lines), len(curr_lines), 1)
    return changed, round(changed / denom, 4)


def verify_real_changes(revision_clones: dict, all_revisions: list,
                        source_root: Path):
    """For every (rev, gcid) with changeType=='M', apply three-level
    verification to decide whether the change is real.

    Returns
    -------
    real_m : dict[rev][gcid] -> bool
    magnitude : dict[rev][gcid] -> float  (0.0 if spurious)
    stats : dict with counters
    """
    gcid_timeline: dict = defaultdict(list)
    for rev in all_revisions:
        for c in revision_clones.get(rev, []):
            gcid_timeline[c["globalCloneId"]].append(
                (rev, c["filePath"], c["startLine"], c["endLine"], c["changeType"]))

    real_m: dict = defaultdict(dict)
    magnitude: dict = defaultdict(dict)
    stats = {
        "m_total": 0, "m_real": 0,
        "spurious_ws": 0,      # same after whitespace norm
        "spurious_comment": 0, # same after comment strip
        "unverifiable": 0,
    }

    for gcid, timeline in gcid_timeline.items():
        for idx, (rev, fp, s, e, ct) in enumerate(timeline):
            if ct != "M":
                continue
            stats["m_total"] += 1

            if idx == 0:
                real_m[rev][gcid] = True
                magnitude[rev][gcid] = 1.0
                stats["m_real"] += 1
                continue

            prev_rev, prev_fp, prev_s, prev_e, _ = timeline[idx - 1]
            curr_lines = _extract_lines(source_root, fp, s, e)
            prev_lines = _extract_lines(source_root, prev_fp, prev_s, prev_e)

            if not curr_lines or not prev_lines:
                real_m[rev][gcid] = True
                magnitude[rev][gcid] = 1.0
                stats["unverifiable"] += 1
                continue

            # Level 1 — whitespace normalization
            curr_ws = _normalize_ws(curr_lines)
            prev_ws = _normalize_ws(prev_lines)
            if curr_ws == prev_ws:
                real_m[rev][gcid] = False
                magnitude[rev][gcid] = 0.0
                stats["spurious_ws"] += 1
                continue

            # Level 2 — comment stripping
            curr_nc = _strip_comments(curr_ws)
            prev_nc = _strip_comments(prev_ws)
            if curr_nc == prev_nc:
                real_m[rev][gcid] = False
                magnitude[rev][gcid] = 0.0
                stats["spurious_comment"] += 1
                continue

            # Level 3 — real change; measure magnitude
            changed_lines, mag = _change_magnitude(prev_lines, curr_lines)
            real_m[rev][gcid] = True
            magnitude[rev][gcid] = mag
            stats["m_real"] += 1

    return real_m, magnitude, stats


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


def build_histories(revision_clones: dict, all_revisions: list,
                    real_m: dict = None):
    """Build FragmentHistory for every globalCloneId.
    When real_m is provided, spurious M events are downgraded to U so
    they are excluded from changeRevs and rev_status."""
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
            if raw_ct == "M" and real_m is not None:
                eff_ct = "M" if real_m.get(rev, {}).get(gcid, True) else "U"
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

    for gcid in active_gcids:
        histories[gcid].deletedInRev = 0

    return histories, class_members, rev_status


# ── FEATURE COMPUTATION ──────────────────────────────────────────────────────

def lambda_from_half_life(h: int) -> float:
    return math.log(2.0) / h


def compute_decay_weighted(change_revs_before: list, target_set: set,
                           current_rev: int, lam: float) -> float:
    w_target = w_total = 0.0
    for r in change_revs_before:
        w = math.exp(-lam * (current_rev - r))
        w_total += w
        if r in target_set:
            w_target += w
    return w_target / w_total if w_total > 1e-9 else 0.0


def compute_features(
    gcid: int, r_t: int, history: FragmentHistory,
    class_members: dict, histories: dict, rev_status: dict,
    clone_info: dict, churn_cache: dict, rev_author: dict, system: str,
    magnitude: dict = None,
) -> dict:
    """All features use ONLY history strictly before r_t."""

    classId = history.classId
    all_peers = class_members.get(classId, set()) - {gcid}
    prior_changes = [r for r in history.changeRevs if r < r_t]
    n_prior = len(prior_changes)

    co_change_revs, indep_revs = set(), set()
    for r in prior_changes:
        rs = rev_status.get(r, {})
        if any(rs.get(p) == "M" for p in all_peers):
            co_change_revs.add(r)
        else:
            indep_revs.add(r)

    n_co    = len(co_change_revs)
    n_indep = len(indep_revs)
    co_ratio = n_co / n_prior if n_prior > 0 else 0.0

    last_change_rev     = max(prior_changes) if prior_changes else history.addedInRev
    revs_since_last     = r_t - last_change_rev
    co_sorted           = sorted(co_change_revs)
    last_co_rev         = co_sorted[-1] if co_sorted else -1
    revs_since_last_co  = (r_t - last_co_rev) if last_co_rev > 0 else -1
    indep_sorted        = sorted(indep_revs)
    last_indep_rev      = indep_sorted[-1] if indep_sorted else -1
    revs_since_last_ind = (r_t - last_indep_rev) if last_indep_rev > 0 else -1
    solo_since_co       = (sum(1 for r in indep_revs if r > last_co_rev)
                           if last_co_rev > 0 else n_indep)

    recent = prior_changes[-RECENT_WINDOW:]
    recent_co_ratio = (sum(1 for r in recent if r in co_change_revs) / len(recent)
                       if recent else 0.0)

    file_age      = r_t - history.addedInRev
    change_pron   = n_prior / file_age if file_age > 0 else 0.0
    total_alive   = max(history.endRev - history.addedInRev + 1, 1)
    prior_unch    = sum(1 for r in history.unchangedRevs if r < r_t)
    stability_idx = prior_unch / total_alive

    alive_peers = [p for p in all_peers
                   if histories[p].addedInRev <= r_t
                   and (histories[p].deletedInRev == 0
                        or histories[p].deletedInRev > r_t)]
    alive_peer_count = len(alive_peers)
    class_size = len(class_members.get(classId, set()))

    frac_same_file = frac_same_method = 0.0
    peer_co_ratios, peer_depths = [], []
    my_fn = extract_filename(history.filePath)
    my_mt = (clone_info.get(gcid, {}) or {}).get("method_name")

    for peer in alive_peers:
        ph = histories[peer]
        if my_fn and extract_filename(ph.filePath) == my_fn:
            frac_same_file += 1
        pm = (clone_info.get(peer, {}) or {}).get("method_name")
        if my_mt and pm and my_mt == pm:
            frac_same_method += 1
        pair_co = sum(1 for r in prior_changes
                      if rev_status.get(r, {}).get(peer) == "M")
        peer_co_ratios.append(pair_co / n_prior if n_prior > 0 else 0.0)
        peer_depths.append(calculate_path_depth(history.filePath, ph.filePath))

    if alive_peer_count > 0:
        frac_same_file /= alive_peer_count
        frac_same_method /= alive_peer_count
    mean_peer_co = float(np.mean(peer_co_ratios)) if peer_co_ratios else 0.0
    max_peer_co  = max(peer_co_ratios) if peer_co_ratios else 0.0
    min_peer_co  = min(peer_co_ratios) if peer_co_ratios else 0.0
    mean_depth   = float(np.mean(peer_depths)) if peer_depths else -1.0

    # Class-level co-change density
    n_active = sum(
        1 for peer in alive_peers
        if any(rev_status.get(r, {}).get(peer) == "M"
               for r in (histories[peer].changeRevs)
               if r < r_t)
    )
    class_co_density = n_active / alive_peer_count if alive_peer_count > 0 else 0.0

    decay_feats = {}
    for hl in HALF_LIVES:
        lam = lambda_from_half_life(hl)
        decay_feats[f"ir_decay_h{hl}"] = compute_decay_weighted(
            prior_changes, indep_revs, r_t, lam)
        decay_feats[f"cs_decay_h{hl}"] = compute_decay_weighted(
            prior_changes, co_change_revs, r_t, lam)

    mid = len(prior_changes) // 2
    if mid >= 2:
        early_co  = sum(1 for r in prior_changes[:mid] if r in co_change_revs) / mid
        recent_co = sum(1 for r in prior_changes[mid:] if r in co_change_revs) / (len(prior_changes) - mid)
        coupling_trend = recent_co - early_co
    else:
        coupling_trend = 0.0

    my_bare = extract_filename(history.filePath)
    total_churn = 0
    for r in prior_changes:
        if r not in churn_cache:
            churn_cache[r] = load_change_info_churn(system, r)
        total_churn += churn_cache[r].get(my_bare, 0)

    author_counts = Counter(rev_author.get(r, "UNKNOWN") for r in prior_changes)
    distinct_authors = len(author_counts)
    major_author_prop = (max(author_counts.values()) / n_prior
                         if n_prior > 0 and author_counts else 0.0)

    # Change-magnitude features (from the verified M at r_t itself)
    mag_at_rt = 0.0
    if magnitude is not None:
        mag_at_rt = magnitude.get(r_t, {}).get(gcid, 0.0)
    # Average magnitude over prior real changes
    avg_prior_mag = 0.0
    if magnitude is not None and prior_changes:
        mags = [magnitude.get(r, {}).get(gcid, 0.0) for r in prior_changes]
        avg_prior_mag = float(np.mean(mags))

    return {
        "n_prior_changes":             n_prior,
        "n_prior_co_changes":          n_co,
        "n_prior_independent":         n_indep,
        "co_change_ratio":             round(co_ratio, 4),
        "revs_since_last_change":      revs_since_last,
        "revs_since_last_co_change":   revs_since_last_co,
        "revs_since_last_independent": revs_since_last_ind,
        "recent_co_ratio_last5":       round(recent_co_ratio, 4),
        "solo_events_since_last_co":   solo_since_co,
        "nlines":                      history.nlines,
        "similarity":                  history.similarity,
        "file_age":                    file_age,
        "change_proneness":            round(change_pron, 4),
        "stability_index":             round(stability_idx, 4),
        "change_magnitude":            round(mag_at_rt, 4),
        "avg_prior_change_magnitude":  round(avg_prior_mag, 4),
        "class_size":                  class_size,
        "alive_peer_count":            alive_peer_count,
        "class_co_change_density":     round(class_co_density, 4),
        "frac_peers_same_file":        round(frac_same_file, 4),
        "frac_peers_same_method":      round(frac_same_method, 4),
        "mean_peer_co_ratio":          round(mean_peer_co, 4),
        "max_peer_co_ratio":           round(max_peer_co, 4),
        "min_peer_co_ratio":           round(min_peer_co, 4),
        "mean_peer_depth":             round(mean_depth, 2),
        **{k: round(v, 4) for k, v in decay_feats.items()},
        "coupling_trend":              round(coupling_trend, 4),
        "churn":                       total_churn,
        "distinct_authors":            distinct_authors,
        "major_author_prop":           round(major_author_prop, 4),
    }


# ── DATASET CONSTRUCTION ─────────────────────────────────────────────────────

def build_dataset(system: str, ctype: str, gran: str,
                  source_root: str = None,
                  bulk_edit_threshold: int = BULK_EDIT_THRESHOLD):
    W = 70
    print("=" * W)
    print(f"  Building fragment dataset: {system} / {ctype} / {gran}")
    print("=" * W)

    if source_root is None:
        source_root = DEFAULT_SOURCE_ROOT_TEMPLATE.format(system=system)
    src_path = Path(source_root)
    verify = src_path.exists()
    print(f"  Source root   : {src_path}  ({'found' if verify else 'NOT FOUND — skipping M verification'})")
    print(f"  Bulk-edit cap : {bulk_edit_threshold} real changes/revision")
    print()

    print("Loading GlobalClones...")
    revision_clones, all_revisions = load_global_clones(system, ctype, gran)
    total_records = sum(len(v) for v in revision_clones.values())
    print(f"  {len(all_revisions)} revisions, {total_records:,} total clone-revision records")

    real_m = None
    magnitude = None
    if verify:
        print("\nVerifying real changes (3-level: whitespace > comments > magnitude)...")
        real_m, magnitude, vstats = verify_real_changes(
            revision_clones, all_revisions, src_path)
        mt = vstats["m_total"]
        if mt > 0:
            print(f"  M events total          : {mt:,}")
            print(f"  Real M (code changed)   : {vstats['m_real']:,}  "
                  f"({vstats['m_real']/mt:.1%})")
            print(f"  Spurious — whitespace   : {vstats['spurious_ws']:,}  "
                  f"({vstats['spurious_ws']/mt:.1%})")
            print(f"  Spurious — comment-only : {vstats['spurious_comment']:,}  "
                  f"({vstats['spurious_comment']/mt:.1%})")
            if vstats["unverifiable"]:
                print(f"  Unverifiable (missing)  : {vstats['unverifiable']:,}")

    print("\nBuilding histories...")
    histories, class_members, rev_status = build_histories(
        revision_clones, all_revisions, real_m)
    print(f"  {len(histories)} unique fragments, {len(class_members)} clone classes")

    print("Loading GlobalCloneInfo...")
    clone_info = load_global_clone_info(system, ctype, gran, all_revisions)
    print(f"  Method info for {len(clone_info)} fragments")

    print("Loading commit log...")
    rev_author = load_commit_log(system)
    print(f"  Author info for {len(rev_author)} revisions")

    churn_cache = {}
    rows = []
    stats = {"total": 0, "dead_peers": 0, "warmup": 0, "emitted": 0}

    changed_revisions = [r for r in all_revisions
                         if any(ct == "M" for ct in rev_status.get(r, {}).values())]

    print(f"\nBuilding prediction instances "
          f"({len(changed_revisions)} changed revisions)...")

    for rev_idx, r_t in enumerate(changed_revisions):
        rs = rev_status.get(r_t, {})
        changed_gcids = [gcid for gcid, ct in rs.items() if ct == "M"]

        for gcid in changed_gcids:
            stats["total"] += 1
            h = histories.get(gcid)
            if h is None:
                continue

            classId = h.classId
            all_peers = class_members.get(classId, set()) - {gcid}
            alive_peers = [p for p in all_peers
                           if histories[p].addedInRev <= r_t
                           and (histories[p].deletedInRev == 0
                                or histories[p].deletedInRev > r_t)]
            if not alive_peers:
                stats["dead_peers"] += 1
                continue

            prior_changes = [r for r in h.changeRevs if r < r_t]
            if len(prior_changes) < MIN_PRIOR_CHANGES:
                stats["warmup"] += 1
                continue

            any_peer_changed = any(rs.get(p) == "M" for p in alive_peers)
            label = 0 if any_peer_changed else 1  # 0=dependent, 1=independent

            features = compute_features(
                gcid, r_t, h, class_members, histories,
                rev_status, clone_info, churn_cache, rev_author, system, magnitude,
            )

            rows.append({
                "Revision": r_t,
                "gcid":     gcid,
                "classid":  classId,
                **features,
                "label":    label,
            })
            stats["emitted"] += 1

        if (rev_idx + 1) % 100 == 0:
            print(f"  ... {rev_idx+1}/{len(changed_revisions)} revisions "
                  f"({stats['emitted']} instances)")

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["Revision", "gcid"]).reset_index(drop=True)

    # Bulk-edit filter
    if bulk_edit_threshold > 0 and not df.empty:
        rev_real_m_count = {
            rev: sum(1 for ct in status.values() if ct == "M")
            for rev, status in rev_status.items()
        }
        bulk_revs = {rev for rev, n in rev_real_m_count.items()
                     if n > bulk_edit_threshold}
        rows_before = len(df)
        df = df[~df["Revision"].isin(bulk_revs)].reset_index(drop=True)
        print(f"\n  Bulk-edit filter: {len(bulk_revs)} revision(s) with "
              f">{bulk_edit_threshold} real changes dropped "
              f"({rows_before - len(df)} rows removed)")

    # Summary
    print(f"\n{'=' * W}")
    print(f"  DATASET SUMMARY: {system} / {ctype} / {gran}")
    print(f"{'=' * W}")
    print(f"  Total fragment change events : {stats['total']:,}")
    print(f"  Excluded (all peers dead)    : {stats['dead_peers']:,}")
    print(f"  Excluded (warm-up < {MIN_PRIOR_CHANGES})      : {stats['warmup']:,}")
    print(f"  Final prediction instances   : {stats['emitted']:,}")

    if not df.empty:
        dep = int((df["label"] == 0).sum())
        ind = int((df["label"] == 1).sum())
        total = dep + ind
        print(f"  Unique fragments             : {df['gcid'].nunique()}")
        print(f"  Unique revisions             : {df['Revision'].nunique()}")
        print(f"  Label: dependent  (0)        : {dep}  ({dep/total:.1%})")
        print(f"  Label: independent (1)       : {ind}  ({ind/total:.1%})")
        print(f"  Revision range               : "
              f"{df['Revision'].min()} - {df['Revision'].max()}")
        print(f"  Features                     : {df.shape[1] - 3}")  # excl Revision/gcid/classid

    out_dir = WORK_FOLDER / system / "Datasets" / "CloneGenealogy"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ctype}_{gran}_fragment_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Saved -> {out_path}")
    print("=" * W)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build fragment-level forecasting dataset from raw GlobalClones data")
    parser.add_argument("--system", default="Ctags")
    parser.add_argument("--type",   default="Type3")
    parser.add_argument("--gran",   default="Block")
    parser.add_argument("--source-root", default=None,
                        help="Root holding Revision_<n>/ source dirs. "
                             "Default: D:/Thesis/NiCad-6.2/systems/{system}")
    parser.add_argument("--bulk-edit-threshold", type=int,
                        default=BULK_EDIT_THRESHOLD,
                        help="Drop revisions with more than N real changes "
                             "(project-wide refactors). 0=disabled. Default=30.")
    args = parser.parse_args()
    build_dataset(args.system, args.type, args.gran,
                  args.source_root, args.bulk_edit_threshold)
