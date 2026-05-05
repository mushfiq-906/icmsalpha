
import os
import subprocess
import sys
import time
import stat
import shutil
import tarfile
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import xml.etree.ElementTree as ET
from xml.dom import minidom

# =====================================================
# CONFIGURATION
# =====================================================
REPO_URL = "https://github.com/jfree/jfreechart.git"
REPO_DIR = "/Volumes/Shafi/Thesis-2.0/jfreechart"
OUTPUT_DIR = "/Volumes/Shafi/Thesis-2.0/Nicad/systems/jfreechart"
ALLOWED_EXTENSIONS = ['.java']

COMMIT_FILE = os.path.join(OUTPUT_DIR, "commits.txt")
FULL_COMMIT_FILE = os.path.join(OUTPUT_DIR, "full_commit.txt")
XML_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "commits.xml")
COMMIT_LOG_FILE = os.path.join(OUTPUT_DIR, "commit_logs.txt")
MAX_WORKERS = 8


# =====================================================
# WINDOWS-SAFE REMOVE
# =====================================================
def force_remove_dir(path):
    def onerror(func, path, exc_info):
        try:
            os.chmod(path, stat.S_IWRITE)
            func(path)
        except:
            pass
    if os.path.exists(path):
        shutil.rmtree(path, onerror=onerror)


# =====================================================
# UTILS
# =====================================================
def run_command(cmd, cwd=None, binary=False):
    try:
        r = subprocess.run(
            cmd, cwd=cwd, shell=True, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        return r.stdout if binary else r.stdout.decode("utf-8", errors="ignore").strip()
    except subprocess.CalledProcessError:
        return None

def should_include_file(path):
    if not ALLOWED_EXTENSIONS:
        return True
    return os.path.splitext(path)[1].lower() in ALLOWED_EXTENSIONS


def get_exact_paths_from_tree(commit_hash):
    """
    Use git ls-tree to get the EXACT canonical paths as stored in the git object tree.
    This is the single source of truth for filenames — avoids any OS-level case folding.
    Returns a dict: lower(path) -> exact_path for quick lookup.
    """
    result = subprocess.run(
        ['git', 'ls-tree', '-r', '--name-only', commit_hash],
        cwd=REPO_DIR, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if result.returncode != 0:
        return {}
    lines = result.stdout.decode('utf-8', errors='ignore').splitlines()
    # Map lowercased path -> exact git-tree path
    return {p.lower(): p for p in lines if should_include_file(p)}


# =====================================================
# GIT SETUP
# =====================================================
def clone_repository():
    if os.path.exists(REPO_DIR):
        if os.path.exists(os.path.join(REPO_DIR, ".git")):
            print("[✓] Repository already exists — reusing it")
            return
        else:
            print(f"[✗] {REPO_DIR} exists but is not a git repository")
            sys.exit(1)

    print("[+] Cloning repository...")
    if run_command(f"git clone --filter=blob:none {REPO_URL} {REPO_DIR}") is None:
        print("[✗] Clone failed")
        sys.exit(1)
    print("[✓] Clone successful")

def get_total_commits():
    count = run_command('git rev-list --count HEAD', cwd=REPO_DIR)
    if count and count.isdigit():
        return int(count)
    return 0

def export_full_commit_history():
    print("\n[+] Exporting full commit history...")
    # Use NUL byte as separator to avoid issues with commit messages containing ' | '
    # or commits with empty messages
    result = subprocess.run(
        ['git', 'log', '--reverse', '--pretty=format:%H%x00%s'],
        cwd=REPO_DIR, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if result.returncode != 0 or not result.stdout.strip():
        print("[✗] Failed to read commit history")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    commit_lines = result.stdout.strip().splitlines()

    with open(FULL_COMMIT_FILE, "w", encoding="utf-8") as f:
        for i, line in enumerate(commit_lines, start=1):
            parts = line.split('\x00', 1)
            commit_hash = parts[0].strip()
            commit_msg = parts[1].strip() if len(parts) > 1 else '(no message)'
            if not commit_msg:
                commit_msg = '(no message)'
            f.write(f"{i} | {commit_hash} | {commit_msg}\n")

    print(f"[✓] Exported {len(commit_lines)} commits to: {FULL_COMMIT_FILE}")

def get_commit_range():
    total_commits = get_total_commits()
    print(f"\n[i] Total commits in repository: {total_commits}")
    print("[i] Commits are numbered from 1 (oldest) to {} (newest)".format(total_commits))
    print(f"[i] Full commit history available in: {FULL_COMMIT_FILE}")
    print("\nEnter the range of commits to extract:")

    while True:
        try:
            start_str = input("  From commit (1-{}): ".format(total_commits)).strip()
            end_str = input("  To commit (1-{}): ".format(total_commits)).strip()
            start = int(start_str)
            end = int(end_str)

            if start < 1 or start > total_commits:
                print(f"  [✗] Start commit must be between 1 and {total_commits}")
                continue
            if end < 1 or end > total_commits:
                print(f"  [✗] End commit must be between 1 and {total_commits}")
                continue
            if start > end:
                print("  [✗] Start commit must be <= end commit")
                continue

            count = end - start + 1
            print(f"\n[✓] Will extract {count} commits (from #{start} to #{end})")
            confirm = input("  Proceed? [Y/n]: ").strip().lower()
            if confirm in ['', 'y', 'yes']:
                return start, end
            else:
                print("  Cancelled. Please enter new range.\n")

        except ValueError:
            print("  [✗] Please enter valid numbers")
        except KeyboardInterrupt:
            print("\n\n[✗] Cancelled by user")
            sys.exit(0)

def generate_commit_list(start, end):
    print(f"\n[+] Reading commits {start} to {end} from {FULL_COMMIT_FILE}...")

    if not os.path.exists(FULL_COMMIT_FILE):
        print("[✗] full_commit.txt not found!")
        sys.exit(1)

    with open(FULL_COMMIT_FILE, "r", encoding="utf-8") as f:
        all_lines = f.readlines()

    commit_lines = []
    for line in all_lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split(' | ', 2)
        if len(parts) < 2:
            continue
        try:
            rev_num = int(parts[0])
            commit_hash = parts[1]
            commit_msg = parts[2] if len(parts) > 2 else '(no message)'
        except (ValueError, IndexError):
            continue
        if start <= rev_num <= end:
            commit_lines.append(f"{commit_hash} | {commit_msg}")

    if not commit_lines:
        print("[✗] No commits found in range")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(COMMIT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(commit_lines))

    print(f"[✓] Found {len(commit_lines)} commits")
    return commit_lines


# =====================================================
# CHANGE DETECTION — uses ls-tree for exact paths
# =====================================================
def get_all_commit_changes(commit_hashes):
    """
    Get added/modified/deleted for each commit.
    All paths are resolved via git ls-tree to guarantee exact canonical casing.
    """
    print("\n[+] Analyzing commit changes (sequential, exact-path resolution)...")
    changes_map = {}

    for i, commit_hash in enumerate(commit_hashes):
        parent_hash = commit_hashes[i - 1] if i > 0 else None

        # Build exact-path lookup for THIS commit
        exact_paths = get_exact_paths_from_tree(commit_hash)

        added, modified, deleted = [], [], []

        if parent_hash is None:
            # First commit — every file is "added"
            added = list(exact_paths.values())
        else:
            # Diff against parent
            output = run_command(
                f'git diff --name-status {parent_hash} {commit_hash}',
                cwd=REPO_DIR
            )

            # Also build parent exact-path lookup for deletes
            parent_exact_paths = get_exact_paths_from_tree(parent_hash)

            if output:
                for line in output.splitlines():
                    if not line.strip():
                        continue
                    parts = line.split('\t', 1)
                    if len(parts) < 2:
                        continue

                    status = parts[0].strip()
                    filepath = parts[1].strip().replace('\\', '/')

                    # Handle renames: R100\told\tnew
                    if status.startswith('R') and '\t' in filepath:
                        old_name, new_name = filepath.split('\t', 1)
                        old_name = old_name.replace('\\', '/')
                        new_name = new_name.replace('\\', '/')
                        # Resolve exact paths
                        exact_old = parent_exact_paths.get(old_name.lower(), old_name)
                        exact_new = exact_paths.get(new_name.lower(), new_name)
                        if should_include_file(exact_old):
                            deleted.append(exact_old)
                        if should_include_file(exact_new):
                            added.append(exact_new)
                        continue

                    if not should_include_file(filepath):
                        continue

                    if status == 'A':
                        # Use exact path from current commit tree
                        exact = exact_paths.get(filepath.lower(), filepath)
                        added.append(exact)
                    elif status == 'M':
                        exact = exact_paths.get(filepath.lower(), filepath)
                        modified.append(exact)
                    elif status == 'D':
                        # Use exact path from parent tree
                        exact = parent_exact_paths.get(filepath.lower(), filepath)
                        deleted.append(exact)
                    elif status.startswith('C'):
                        exact = exact_paths.get(filepath.lower(), filepath)
                        added.append(exact)

        changes_map[commit_hash] = (added, modified, deleted)
        change_count = len(added) + len(modified) + len(deleted)
        print(f"  [{i+1}/{len(commit_hashes)}] {commit_hash[:10]} - {change_count} changes")

    print("[✓] Change analysis complete\n")
    return changes_map


# =====================================================
# EXTRACTION — uses git ls-tree members for exact paths
# =====================================================
def extract_commit_with_archive(commit_hash, rev_dir):
    """
    Extract filtered files from git archive.
    After extraction, rename any files whose on-disk name differs from the
    canonical git-tree path (fixes Windows case-folding).
    """
    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp(prefix="git_extract_")
        tar_path = os.path.join(temp_dir, "snapshot.tar")

        result = run_command(
            f'git archive --format=tar --output="{tar_path}" {commit_hash}',
            cwd=REPO_DIR
        )
        if result is None or not os.path.exists(tar_path):
            return []

        # Get the canonical paths from git tree BEFORE extracting
        exact_paths = get_exact_paths_from_tree(commit_hash)

        extracted_files = []

        with tarfile.open(tar_path, 'r') as tar:
            for member in tar.getmembers():
                if member.isdir():
                    continue

                member_path = member.name.replace('\\', '/')
                if not should_include_file(member_path):
                    continue

                # Resolve to canonical path from git tree
                canonical = exact_paths.get(member_path.lower(), member_path)

                dest_path = os.path.join(rev_dir, canonical)
                dest_dir = os.path.dirname(dest_path)
                os.makedirs(dest_dir, exist_ok=True)

                file_obj = tar.extractfile(member)
                if file_obj:
                    with open(dest_path, 'wb') as f:
                        f.write(file_obj.read())
                    extracted_files.append(canonical)

        return extracted_files

    except Exception as e:
        print(f"Error extracting archive for {commit_hash}: {e}")
        return []
    finally:
        if temp_dir and os.path.exists(temp_dir):
            force_remove_dir(temp_dir)


# =====================================================
# COMMIT PROCESSING (THREAD WORK)
# =====================================================
def process_commit_extraction(rev_num, total_revs, commit_hash, commit_msg):
    rev_dir = os.path.join(OUTPUT_DIR, f"Revision_{rev_num}")
    force_remove_dir(rev_dir)
    os.makedirs(rev_dir, exist_ok=True)

    print(f"[r{rev_num}] Extracting {commit_hash[:10]} | {commit_msg[:50]}")

    files = extract_commit_with_archive(commit_hash, rev_dir)

    if not files:
        print(f"[!] r{rev_num} - No files extracted")
    else:
        print(f"[✓] r{rev_num} - Extracted {len(files)} files")

    return rev_num


# =====================================================
# PARALLEL EXECUTION
# =====================================================
def process_commits_parallel(commits, changes_map, start_commit):
    total = len(commits)

    commit_data = []
    for c in commits:
        parts = c.split('|', 1)
        commit_hash = parts[0].strip()
        commit_msg = parts[1].strip() if len(parts) > 1 else ''
        commit_data.append((commit_hash, commit_msg))

    print(f"\n[+] Extracting {total} commits with {MAX_WORKERS} worker threads...\n")

    completed = set()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for i in range(total):
            commit_hash, commit_msg = commit_data[i]
            actual_rev_num = start_commit + i
            future = executor.submit(
                process_commit_extraction,
                actual_rev_num, total, commit_hash, commit_msg
            )
            futures[future] = actual_rev_num

        for future in as_completed(futures):
            try:
                rev_num = future.result()
                completed.add(rev_num)
            except Exception as e:
                print(f"[✗] Error processing commit: {e}")

    results = {}
    for i, (commit_hash, _) in enumerate(commit_data):
        actual_rev_num = start_commit + i
        if actual_rev_num in completed:
            added, modified, deleted = changes_map.get(commit_hash, ([], [], []))
            results[actual_rev_num] = (added, modified, deleted)

    return results


# =====================================================
# XML WRITE
# =====================================================
def write_sorted_xml(results):
    print("\n[+] Writing XML file...")
    root = ET.Element("commits")

    for rev_id in sorted(results.keys()):
        added, modified, deleted = results[rev_id]
        rev = ET.SubElement(root, "revision", id=str(rev_id))

        if added:
            a = ET.SubElement(rev, "added")
            for f in sorted(added):
                ET.SubElement(a, "file", path=f)
        if modified:
            m = ET.SubElement(rev, "modified")
            for f in sorted(modified):
                ET.SubElement(m, "file", path=f)
        if deleted:
            d = ET.SubElement(rev, "deleted")
            for f in sorted(deleted):
                ET.SubElement(d, "file", path=f)

    pretty = minidom.parseString(
        ET.tostring(root, encoding="utf-8")
    ).toprettyxml(indent="  ")

    with open(XML_OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(pretty)

    print(f"[✓] XML written to: {XML_OUTPUT_FILE}")


# =====================================================
# COMMIT_LOGS.TXT (ICMS Alpha compatible)
# =====================================================
def generate_commit_logs(commits, changes_map, start_commit):
    """
    Generate commit_logs.txt in the pipe-delimited format used by ICMS Alpha.
    Format: revision | filePath | changeType | isBugFix | commitMessage | author | date

    This file is read by AccessPoint.getSVNChanges() in the Java tool.
    """
    print("\n[+] Generating commit_logs.txt (ICMS Alpha format)...")

    # We need author and date info — re-read from git log
    commit_data = []
    for c in commits:
        parts = c.split('|', 1)
        commit_hash = parts[0].strip()
        commit_msg = parts[1].strip() if len(parts) > 1 else ''
        commit_data.append((commit_hash, commit_msg))

    # Get full metadata for each commit (author + date)
    commit_meta = {}
    for commit_hash, _ in commit_data:
        result = subprocess.run(
            ['git', 'log', '-1', '--format=%an|%ai', commit_hash],
            cwd=REPO_DIR, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if result.returncode == 0 and result.stdout.strip():
            meta_parts = result.stdout.strip().split('|', 1)
            author = meta_parts[0] if len(meta_parts) > 0 else 'unknown'
            date = meta_parts[1] if len(meta_parts) > 1 else ''
            commit_meta[commit_hash] = (author, date)
        else:
            commit_meta[commit_hash] = ('unknown', '')

    with open(COMMIT_LOG_FILE, 'w', encoding='utf-8') as f:
        # Header (matches SVN format)
        f.write("revision | filePath | changeType | isBugFix | commitMessage | author | date\n")

        for i, (commit_hash, commit_msg) in enumerate(commit_data):
            rev_num = start_commit + i
            author, date = commit_meta.get(commit_hash, ('unknown', ''))

            # Escape pipes in commit message
            safe_msg = commit_msg.replace('|', '-')

            # Bug fix heuristic
            is_bug_fix = is_bug_fix_commit(safe_msg)

            # Get changed files from the pre-computed changes_map
            added, modified, deleted = changes_map.get(commit_hash, ([], [], []))

            if not added and not modified and not deleted:
                # Write a single entry even if no file changes
                file_path = f"Revision_{rev_num}/"
                f.write(f"{rev_num} | {file_path} | A | {is_bug_fix} | {safe_msg} | {author} | {date}\n")
            else:
                for file_name in added:
                    file_path = f"Revision_{rev_num}/{file_name}"
                    f.write(f"{rev_num} | {file_path} | A | {is_bug_fix} | {safe_msg} | {author} | {date}\n")
                for file_name in modified:
                    file_path = f"Revision_{rev_num}/{file_name}"
                    f.write(f"{rev_num} | {file_path} | M | {is_bug_fix} | {safe_msg} | {author} | {date}\n")
                for file_name in deleted:
                    file_path = f"Revision_{rev_num}/{file_name}"
                    f.write(f"{rev_num} | {file_path} | D | {is_bug_fix} | {safe_msg} | {author} | {date}\n")

    print(f"[✓] commit_logs.txt written to: {COMMIT_LOG_FILE}")


def is_bug_fix_commit(message):
    """Heuristic to detect bug fix commits from the commit message."""
    if not message:
        return "no"
    lower = message.lower()
    keywords = ['fix', 'bug', 'patch', 'issue', 'error', 'defect',
                'resolve', 'repair', 'crash', 'fault', 'failure',
                'correct', 'hotfix']
    return "yes" if any(kw in lower for kw in keywords) else "no"


# =====================================================
# MAIN
# =====================================================
def main():
    print("\n" + "="*60)
    print("  GIT → FULL SNAPSHOT EXTRACTOR")
    print("="*60)
    print(f"\nRepository: {REPO_URL}")
    print(f"Output directory: {OUTPUT_DIR}/")
    print(f"File filter: {ALLOWED_EXTENSIONS}")
    print(f"Workers: {MAX_WORKERS} threads")
    print("="*60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    clone_repository()
    export_full_commit_history()

    start, end = get_commit_range()
    commits = generate_commit_list(start, end)

    if not commits:
        print("[✗] No commits found")
        sys.exit(1)

    commit_hashes = [c.split('|', 1)[0].strip() for c in commits]

    start_time = time.time()

    changes_map = get_all_commit_changes(commit_hashes)
    results = process_commits_parallel(commits, changes_map, start)
    write_sorted_xml(results)
    generate_commit_logs(commits, changes_map, start)

    elapsed = time.time() - start_time

    total_added   = sum(len(a) for a, _, _ in results.values())
    total_modified= sum(len(m) for _, m, _ in results.values())
    total_deleted = sum(len(d) for _, _, d in results.values())

    print("\n" + "="*60)
    print("  EXTRACTION COMPLETE")
    print("="*60)
    print(f"✓ Processed: {len(results)} revisions (commits {start} to {end})")
    print(f"✓ Added: {total_added}  Modified: {total_modified}  Deleted: {total_deleted}")
    print(f"✓ Time: {elapsed:.2f}s  Speed: {len(results)/elapsed:.2f} commits/sec")
    print(f"✓ Output: {os.path.abspath(OUTPUT_DIR)}/")
    print(f"✓ commit_logs.txt: {os.path.abspath(COMMIT_LOG_FILE)}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()