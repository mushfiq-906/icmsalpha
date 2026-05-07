"""
Diagnose WHY the fragment-level data shows so many "dependent" labels.
The hypothesis: too many false "M" (modified) changeTypes due to line shifts.
When a file is modified (e.g., lines inserted at top), ALL clone fragments 
in that file appear "modified" because their line ranges shift, even though
the clone's actual source code didn't change.
"""
import os
from pathlib import Path
from collections import defaultdict, Counter

WORK_FOLDER = Path(__file__).resolve().parent.parent / "WorkFolder"

def analyze_system(system, ctype="Type3", gran="Block"):
    clone_dir = WORK_FOLDER / system / "Clones" / ctype / gran / "GlobalClones"
    
    # Parse all revisions
    rev_data = {}  # rev -> list of {gcid, classId, changeType, filePath}
    for f in sorted(clone_dir.glob("*_clones.txt")):
        rev = int(f.stem.replace("_clones", ""))
        clones = []
        with open(f, "r", encoding="latin-1") as fh:
            fh.readline()  # skip header
            for line in fh:
                parts = line.strip().split("|")
                if len(parts) < 9:
                    continue
                try:
                    clones.append({
                        "gcid": int(parts[8].strip()),
                        "classId": int(parts[7].strip()),
                        "changeType": parts[2].strip(),
                        "filePath": parts[1].strip(),
                    })
                except (ValueError, IndexError):
                    continue
        rev_data[rev] = clones
    
    print(f"\n{'='*70}")
    print(f"  {system}: Change Type Analysis")
    print(f"{'='*70}")
    
    # 1. Overall changeType distribution across all revision-fragment pairs
    type_counts = Counter()
    for rev, clones in rev_data.items():
        for c in clones:
            type_counts[c["changeType"]] += 1
    total = sum(type_counts.values())
    print(f"\nOverall changeType distribution:")
    for ct in sorted(type_counts.keys()):
        print(f"  {ct}: {type_counts[ct]:>8,} ({type_counts[ct]/total:.1%})")
    
    # 2. Per-revision: what fraction of fragments have M?
    rev_m_fracs = []
    for rev in sorted(rev_data.keys()):
        clones = rev_data[rev]
        if len(clones) == 0:
            continue
        n_m = sum(1 for c in clones if c["changeType"] == "M")
        if n_m > 0:
            frac = n_m / len(clones)
            rev_m_fracs.append((rev, n_m, len(clones), frac))
    
    if rev_m_fracs:
        fracs = [x[3] for x in rev_m_fracs]
        print(f"\nPer changed-revision: fraction of fragments with changeType=M:")
        print(f"  Mean:   {sum(fracs)/len(fracs):.1%}")
        print(f"  Median: {sorted(fracs)[len(fracs)//2]:.1%}")
        print(f"  Min:    {min(fracs):.1%} (rev {min(rev_m_fracs, key=lambda x:x[3])[0]})")
        print(f"  Max:    {max(fracs):.1%} (rev {max(rev_m_fracs, key=lambda x:x[3])[0]})")
    
    # 3. File-level analysis: when a file is changed, how many clones in it get M?
    file_analysis = defaultdict(list)  # (rev, file) -> list of changeTypes
    for rev, clones in rev_data.items():
        file_clones = defaultdict(list)
        for c in clones:
            # Extract bare filename
            fp = c["filePath"].replace("\\", "/").split("/")[-1]
            file_clones[fp].append(c["changeType"])
        for fp, types in file_clones.items():
            if "M" in types:
                file_analysis[(rev, fp)] = types
    
    if file_analysis:
        # For files with at least one M, what fraction of clones in that file are M?
        file_m_fracs = []
        for (rev, fp), types in file_analysis.items():
            n_m = types.count("M")
            frac = n_m / len(types)
            file_m_fracs.append(frac)
        
        print(f"\nWhen a file has >=1 modified clone, fraction of ALL clones in that file with M:")
        print(f"  Mean:   {sum(file_m_fracs)/len(file_m_fracs):.1%}")
        print(f"  Median: {sorted(file_m_fracs)[len(file_m_fracs)//2]:.1%}")
        all_m = sum(1 for f in file_m_fracs if f == 1.0)
        print(f"  100% (ALL clones in file got M): {all_m}/{len(file_m_fracs)} ({all_m/len(file_m_fracs):.1%})")
    
    # 4. Co-change analysis: at revisions where fragments change, 
    #    how many unique classes are involved vs. how many fragments?
    print(f"\nPer changed-revision breakdown:")
    rev_details = []
    for rev in sorted(rev_data.keys()):
        clones = rev_data[rev]
        changed = [c for c in clones if c["changeType"] == "M"]
        if not changed:
            continue
        classes_touched = set(c["classId"] for c in changed)
        files_touched = set(c["filePath"].replace("\\","/").split("/")[-1] for c in changed)
        rev_details.append({
            "rev": rev,
            "n_changed": len(changed),
            "n_classes": len(classes_touched),
            "n_files": len(files_touched),
        })
    
    if rev_details:
        avg_changed = sum(d["n_changed"] for d in rev_details) / len(rev_details)
        avg_classes = sum(d["n_classes"] for d in rev_details) / len(rev_details)
        avg_files = sum(d["n_files"] for d in rev_details) / len(rev_details)
        print(f"  Avg fragments changing per revision: {avg_changed:.1f}")
        print(f"  Avg clone classes involved:          {avg_classes:.1f}")
        print(f"  Avg files involved:                  {avg_files:.1f}")
        
        # How many revisions have fragments from same class co-changing?
        n_cochange_revs = sum(1 for d in rev_details if d["n_changed"] > d["n_classes"])
        print(f"  Revisions with within-class co-changes: {n_cochange_revs}/{len(rev_details)} ({n_cochange_revs/len(rev_details):.1%})")
        
        # How many revisions have ALL changes in a single file?
        single_file = sum(1 for d in rev_details if d["n_files"] == 1)
        print(f"  Revisions with changes in single file:  {single_file}/{len(rev_details)}")

    # 5. KEY TEST: For each fragment that changes, how many of its CLASS PEERS
    #    are in the SAME FILE? (If they're in the same file, they'll likely all get M)
    class_members = defaultdict(set)
    gcid_file = {}
    for rev, clones in rev_data.items():
        for c in clones:
            class_members[c["classId"]].add(c["gcid"])
            gcid_file[c["gcid"]] = c["filePath"].replace("\\","/").split("/")[-1]
    
    same_file_fracs = []
    for classId, members in class_members.items():
        if len(members) < 2:
            continue
        files = [gcid_file.get(m, "") for m in members]
        file_counts = Counter(files)
        # What fraction of the class is in the most common file?
        max_same = max(file_counts.values())
        same_file_fracs.append(max_same / len(members))
    
    if same_file_fracs:
        print(f"\nClone class same-file concentration:")
        print(f"  Mean fraction of class in most-common file: {sum(same_file_fracs)/len(same_file_fracs):.1%}")
        all_same = sum(1 for f in same_file_fracs if f == 1.0)
        print(f"  Classes where ALL members are in same file:  {all_same}/{len(same_file_fracs)}")

    # 6. CRITICAL: Check for "false M" - fragments where changeType=M but 
    # the actual clone class peer analysis shows the change affected the WHOLE file
    # Proxy: at a revision, if ALL clones in a file got M, it might be a bulk change
    bulk_m_count = 0
    real_m_count = 0
    for (rev, fp), types in file_analysis.items():
        n_m = types.count("M")
        if n_m == len(types) and len(types) > 1:
            bulk_m_count += n_m
        else:
            real_m_count += n_m
    total_m = bulk_m_count + real_m_count
    if total_m > 0:
        print(f"\nPotential false M analysis:")
        print(f"  'Bulk M' (ALL clones in file got M, >1 clone): {bulk_m_count} ({bulk_m_count/total_m:.1%})")
        print(f"  'Selective M' (not all clones in file got M):   {real_m_count} ({real_m_count/total_m:.1%})")
        print(f"  NOTE: Bulk M suggests line-shift artifacts, not genuine fragment modifications")

for sys in ["Ctags", "Jmol"]:
    analyze_system(sys)
