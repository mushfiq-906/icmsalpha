import subprocess
import os
from multiprocessing import Pool, cpu_count

repo_url = "svn://svn.code.sf.net/p/ctags/code/trunk"
start_rev = 1
end_rev   = 200
workers = 20  # as you asked

def download_revision(rev):
    target_dir = f"./rev_{rev}"
    os.makedirs(target_dir, exist_ok=True)

    cmd = [
        "svn", "export",
        "-r", str(rev),
        repo_url,
        target_dir,
        "--force"
    ]

    print(f"⬇️ Downloading r{rev} ...")
    
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode == 0:
        print(f"✅ Completed r{rev}")
    else:
        print(f"❌ Failed r{rev}\n{result.stderr}")

if __name__ == "__main__":
    print(f"Starting export from r{start_rev} to r{end_rev} with {workers} processes...")

    with Pool(processes=workers) as pool:
        pool.map(download_revision, range(start_rev, end_rev + 1))

    print("🎉 All revisions done.")
