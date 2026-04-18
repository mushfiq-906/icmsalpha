# import subprocess
# from pathlib import Path
# import xml.etree.ElementTree as ET
# import tkinter as tk
# from tkinter import ttk, scrolledtext, messagebox
# import threading
# import queue
# import time

# MAX_THREADS = 5
# MAX_RETRIES = 3

# # ---------------- SVN Commit Fetch ----------------
# def fetch_commit_log(rev, repo_url, extensions_set, log_queue, results):
#     """Fetch a single revision commit log with retries."""
#     for attempt in range(1, MAX_RETRIES+1):
#         try:
#             cmd = ["svn", "log", "-r", str(rev), repo_url, "--verbose", "--xml"]
#             result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#             if result.returncode != 0:
#                 raise RuntimeError(result.stderr.strip())

#             root = ET.fromstring(result.stdout)
#             entries = []
#             for logentry in root.findall('logentry'):
#                 revision = logentry.attrib['revision']
#                 author = logentry.findtext('author', default='N/A')
#                 date = logentry.findtext('date', default='N/A')
#                 msg = logentry.findtext('msg', default='').replace("\n", " ").strip()
#                 paths = logentry.find('paths')
#                 if paths is not None:
#                     for path in paths.findall('path'):
#                         change_type = path.attrib.get('action', 'M')
#                         filepath = path.text or ''
#                         # Remove /trunk/ prefix
#                         if filepath.startswith("/trunk/"):
#                             filepath = filepath[len("/trunk/"):]
#                         # Filter by extensions
#                         if extensions_set:
#                             if not any(filepath.endswith(ext) for ext in extensions_set):
#                                 continue
#                         line_number = "N/A"  # SVN doesn't provide line numbers
#                         entries.append(f"{revision} | {filepath} | {change_type} | {line_number} | {msg} | {author} | {date}")
#             results.extend(entries)
#             log_queue.put(f"[Success] Revision {rev} fetched")
#             return
#         except Exception as e:
#             log_queue.put(f"[Error] Revision {rev}, attempt {attempt}: {e}")
#             time.sleep(1)
#     log_queue.put(f"[Failed] Revision {rev} after {MAX_RETRIES} attempts")

# # ---------------- GUI ----------------
# class SVNCommitLoggerGUI:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("SVN Commit Log Downloader")
#         self.root.geometry("900x650")
#         self.is_running = False
#         self.log_queue = queue.Queue()
#         self.results = []

#         self.setup_widgets()
#         self.update_logs()

#     # ---------------- Widgets ----------------
#     def setup_widgets(self):
#         frame = ttk.Frame(self.root, padding=10)
#         frame.pack(fill=tk.BOTH, expand=True)

#         # Repository URL
#         ttk.Label(frame, text="SVN Repository URL:").pack(anchor=tk.W)
#         self.repo_var = tk.StringVar(value="https://svn.code.sf.net/p/ctags/code/trunk")
#         ttk.Entry(frame, textvariable=self.repo_var, width=80).pack(anchor=tk.W, pady=(0,10))

#         # Output folder
#         ttk.Label(frame, text="Output Folder:").pack(anchor=tk.W)
#         self.output_var = tk.StringVar(value="./commit_logs")
#         ttk.Entry(frame, textvariable=self.output_var, width=50).pack(anchor=tk.W, pady=(0,10))

#         # File extensions
#         ttk.Label(frame, text="File Extensions (comma-separated, leave empty for all):").pack(anchor=tk.W)
#         self.ext_var = tk.StringVar(value=".c,.h,.cpp,.py,.java")
#         ttk.Entry(frame, textvariable=self.ext_var, width=50).pack(anchor=tk.W, pady=(0,10))

#         # Start / End revision
#         rev_frame = ttk.Frame(frame)
#         rev_frame.pack(anchor=tk.W, pady=(0,10))
#         ttk.Label(rev_frame, text="Start Revision:").pack(side=tk.LEFT)
#         self.start_rev_var = tk.StringVar(value="1")
#         ttk.Entry(rev_frame, textvariable=self.start_rev_var, width=10).pack(side=tk.LEFT, padx=(0,10))
#         ttk.Label(rev_frame, text="End Revision:").pack(side=tk.LEFT)
#         self.end_rev_var = tk.StringVar(value="10")
#         ttk.Entry(rev_frame, textvariable=self.end_rev_var, width=10).pack(side=tk.LEFT, padx=(0,10))

#         # Control buttons
#         btn_frame = ttk.Frame(frame)
#         btn_frame.pack(anchor=tk.W, pady=(0,10))
#         self.start_btn = ttk.Button(btn_frame, text="Start", command=self.start)
#         self.start_btn.pack(side=tk.LEFT, padx=(0,10))
#         self.stop_btn = ttk.Button(btn_frame, text="Stop", command=self.stop, state=tk.DISABLED)
#         self.stop_btn.pack(side=tk.LEFT)

#         # Log area
#         ttk.Label(frame, text="Activity Log:").pack(anchor=tk.W)
#         self.log_text = scrolledtext.ScrolledText(frame, height=25)
#         self.log_text.pack(fill=tk.BOTH, expand=True)

#     # ---------------- Logging ----------------
#     def log(self, message):
#         self.log_queue.put(message)

#     def update_logs(self):
#         try:
#             while True:
#                 message = self.log_queue.get_nowait()
#                 self.log_text.insert(tk.END, message + "\n")
#                 self.log_text.see(tk.END)
#         except queue.Empty:
#             pass
#         self.root.after(100, self.update_logs)

#     # ---------------- Control ----------------
#     def start(self):
#         if self.is_running:
#             return
#         try:
#             self.repo_url = self.repo_var.get().strip()
#             self.output_dir = Path(self.output_var.get().strip())
#             self.start_rev = int(self.start_rev_var.get())
#             self.end_rev = int(self.end_rev_var.get())
#             self.extensions_set = {ext.strip() for ext in self.ext_var.get().split(',') if ext.strip()}
#             if not self.repo_url:
#                 raise ValueError("Repository URL cannot be empty")
#             if self.start_rev > self.end_rev:
#                 raise ValueError("Start revision must be <= End revision")
#         except ValueError as e:
#             messagebox.showerror("Invalid Input", str(e))
#             return

#         self.is_running = True
#         self.start_btn.config(state=tk.DISABLED)
#         self.stop_btn.config(state=tk.NORMAL)
#         self.results.clear()
#         self.thread = threading.Thread(target=self.run)
#         self.thread.daemon = True
#         self.thread.start()

#     def stop(self):
#         self.is_running = False
#         self.log("Stopping download...")
#         self.stop_btn.config(state=tk.DISABLED)

#     # ---------------- Download Logic ----------------
#     def run(self):
#         sem = threading.Semaphore(MAX_THREADS)
#         threads = []

#         def worker(rev):
#             if not self.is_running:
#                 return
#             with sem:
#                 fetch_commit_log(rev, self.repo_url, self.extensions_set, self.log_queue, self.results)
#                 time.sleep(0.1)  # slight delay to avoid overload

#         for rev in range(self.start_rev, self.end_rev + 1):
#             if not self.is_running:
#                 break
#             t = threading.Thread(target=worker, args=(rev,))
#             t.start()
#             threads.append(t)

#         for t in threads:
#             t.join()

#         # Save merged logs
#         if self.results:
#             self.results.sort(key=lambda x: int(x.split(" | ")[0]))  # sort by revision
#             repo_name = self.repo_url.rstrip('/').split('/')[-1] or "repo"
#             self.output_dir.mkdir(parents=True, exist_ok=True)
#             merged_file = self.output_dir / f"{repo_name}_commit_logs.txt"
#             with open(merged_file, "w", encoding="utf-8") as f:
#                 f.write("revision | filepath | change type | line number | commit message | author | date\n")
#                 for line in self.results:
#                     f.write(line + "\n")
#             self.log(f"All revisions merged into {merged_file}")

#         self.is_running = False
#         self.start_btn.config(state=tk.NORMAL)
#         self.stop_btn.config(state=tk.DISABLED)
#         self.log("All done!")

# # ---------------- Main ----------------
# def main():
#     root = tk.Tk()
#     app = SVNCommitLoggerGUI(root)
#     root.mainloop()

# if __name__ == "__main__":
#     main()


import subprocess
from pathlib import Path
import xml.etree.ElementTree as ET
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import queue
from urllib.parse import urlparse
import time

# ---------------- Helper Functions ----------------
def extract_repo_name(svn_url: str) -> str:
    parsed = urlparse(svn_url)
    path_parts = Path(parsed.path).parts
    ignore = {"trunk", "branches", "tags", ""}
    for part in reversed(path_parts):
        if part not in ignore:
            return part
    return "svn_repo"

def test_connection(repo_url, log_queue, timeout=10):
    """Test SVN repository connection."""
    try:
        log_queue.put(f"Testing connection to {repo_url}...")
        cmd = ["svn", "info", repo_url, "--xml"]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                              text=True, timeout=timeout)
        if result.returncode != 0:
            error_msg = result.stderr.strip()
            if "Unable to connect" in error_msg or "Connection" in error_msg:
                return False, f"Network error: Cannot connect to repository"
            elif "timed out" in error_msg.lower():
                return False, f"Network error: Connection timed out"
            else:
                return False, f"Repository error: {error_msg}"
        log_queue.put("Connection successful!")
        return True, None
    except subprocess.TimeoutExpired:
        return False, "Network error: Connection timed out"
    except FileNotFoundError:
        return False, "SVN command not found. Please install Subversion."
    except Exception as e:
        return False, f"Connection test failed: {str(e)}"

def fetch_commit_log(rev, repo_url, extensions_set, log_queue, results_list, 
                     timeout=30, max_retries=3, retry_delay=2):
    """Fetch commit log for a single revision with retry mechanism."""
    for attempt in range(max_retries):
        try:
            cmd = ["svn", "log", "-r", str(rev), repo_url, "--verbose", "--xml"]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                  text=True, timeout=timeout)
            
            if result.returncode != 0:
                error_msg = result.stderr.strip()
                
                # Check for network-related errors
                if any(keyword in error_msg.lower() for keyword in 
                       ["unable to connect", "connection", "timed out", "network", 
                        "host not found", "could not resolve"]):
                    if attempt < max_retries - 1:
                        log_queue.put(f"[Network Error] Revision {rev} (Attempt {attempt+1}/{max_retries}): Retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        log_queue.put(f"[Network Error] Revision {rev}: Failed after {max_retries} attempts - {error_msg}")
                        return False
                else:
                    log_queue.put(f"[Error] Revision {rev}: {error_msg}")
                    return False

            root = ET.fromstring(result.stdout)
            logentry_found = False
            
            for logentry in root.findall('logentry'):
                logentry_found = True
                revision = logentry.attrib['revision']
                author = logentry.findtext('author', default='N/A')
                date = logentry.findtext('date', default='N/A')
                msg = logentry.findtext('msg', default='').replace("\n", " ").strip()
                paths = logentry.find('paths')
                
                if paths is not None:
                    for path in paths.findall('path'):
                        change_type = path.attrib.get('action', 'M')
                        filepath = path.text or ""
                        
                        # Remove /trunk/ prefix
                        if filepath.startswith("/trunk/"):
                            filepath = filepath[len("/trunk/"):]
                        elif filepath.startswith("/trunk"):
                            filepath = filepath[len("/trunk"):]
                        
                        # Check extension filter
                        if extensions_set:
                            ext = Path(filepath).suffix.lower()
                            if ext not in extensions_set:
                                continue
                        
                        line_number = "N/A"
                        results_list.append((int(revision), filepath, change_type, 
                                           line_number, msg, author, date))
            
            if logentry_found:
                log_queue.put(f"[Success] Revision {rev} fetched")
            else:
                log_queue.put(f"[Info] Revision {rev} has no matching entries")
            
            return True
            
        except subprocess.TimeoutExpired:
            if attempt < max_retries - 1:
                log_queue.put(f"[Timeout] Revision {rev} (Attempt {attempt+1}/{max_retries}): Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                continue
            else:
                log_queue.put(f"[Timeout] Revision {rev}: Failed after {max_retries} attempts")
                return False
                
        except ET.ParseError as e:
            log_queue.put(f"[Parse Error] Revision {rev}: Invalid XML response - {e}")
            return False
            
        except Exception as e:
            if attempt < max_retries - 1:
                log_queue.put(f"[Error] Revision {rev} (Attempt {attempt+1}/{max_retries}): {e} - Retrying...")
                time.sleep(retry_delay)
                continue
            else:
                log_queue.put(f"[Error] Revision {rev}: {e}")
                return False
    
    return False

# ---------------- GUI ----------------
class SVNCommitLoggerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SVN Commit Log Downloader")
        self.root.geometry("850x700")
        self.is_running = False
        self.log_queue = queue.Queue()
        self.results = []
        self.setup_widgets()
        self.update_logs()

    def setup_widgets(self):
        frame = ttk.Frame(self.root, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        # Repository URL
        ttk.Label(frame, text="SVN Repository URL:").pack(anchor=tk.W)
        self.repo_var = tk.StringVar(value="svn://svn.code.sf.net/p/ctags/code/trunk")
        ttk.Entry(frame, textvariable=self.repo_var, width=80).pack(anchor=tk.W, pady=(0,10))

        # Output folder
        ttk.Label(frame, text="Output Folder:").pack(anchor=tk.W)
        self.output_var = tk.StringVar(value="./commit_logs")
        ttk.Entry(frame, textvariable=self.output_var, width=50).pack(anchor=tk.W, pady=(0,10))

        # File extensions
        ttk.Label(frame, text="File Extensions (comma-separated, optional):").pack(anchor=tk.W)
        self.ext_var = tk.StringVar(value=".c,.h,.cpp,.py,.java")
        ttk.Entry(frame, textvariable=self.ext_var, width=50).pack(anchor=tk.W, pady=(0,10))

        # Start / End revision
        rev_frame = ttk.Frame(frame)
        rev_frame.pack(anchor=tk.W, pady=(0,10))
        ttk.Label(rev_frame, text="Start Revision:").pack(side=tk.LEFT)
        self.start_rev_var = tk.StringVar(value="1")
        ttk.Entry(rev_frame, textvariable=self.start_rev_var, width=10).pack(side=tk.LEFT, padx=(0,10))
        ttk.Label(rev_frame, text="End Revision:").pack(side=tk.LEFT)
        self.end_rev_var = tk.StringVar(value="10")
        ttk.Entry(rev_frame, textvariable=self.end_rev_var, width=10).pack(side=tk.LEFT, padx=(0,10))

        # Network settings
        net_frame = ttk.Frame(frame)
        net_frame.pack(anchor=tk.W, pady=(0,10))
        ttk.Label(net_frame, text="Timeout (seconds):").pack(side=tk.LEFT)
        self.timeout_var = tk.StringVar(value="30")
        ttk.Entry(net_frame, textvariable=self.timeout_var, width=5).pack(side=tk.LEFT, padx=(5,15))
        ttk.Label(net_frame, text="Max Retries:").pack(side=tk.LEFT)
        self.retry_var = tk.StringVar(value="3")
        ttk.Entry(net_frame, textvariable=self.retry_var, width=5).pack(side=tk.LEFT, padx=(5,15))
        ttk.Label(net_frame, text="Concurrent Threads:").pack(side=tk.LEFT)
        self.threads_var = tk.StringVar(value="5")
        ttk.Entry(net_frame, textvariable=self.threads_var, width=5).pack(side=tk.LEFT, padx=(5,0))

        # Control buttons
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(anchor=tk.W, pady=(0,10))
        self.test_btn = ttk.Button(btn_frame, text="Test Connection", command=self.test_connection)
        self.test_btn.pack(side=tk.LEFT, padx=(0,10))
        self.start_btn = ttk.Button(btn_frame, text="Start", command=self.start)
        self.start_btn.pack(side=tk.LEFT, padx=(0,10))
        self.stop_btn = ttk.Button(btn_frame, text="Stop", command=self.stop, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT)

        # Progress
        self.progress_var = tk.StringVar(value="Ready")
        ttk.Label(frame, textvariable=self.progress_var).pack(anchor=tk.W, pady=(0,5))

        # Log area
        ttk.Label(frame, text="Activity Log:").pack(anchor=tk.W)
        self.log_text = scrolledtext.ScrolledText(frame, height=20)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def log(self, message):
        self.log_queue.put(message)

    def update_logs(self):
        try:
            while True:
                message = self.log_queue.get_nowait()
                self.log_text.insert(tk.END, message + "\n")
                self.log_text.see(tk.END)
        except queue.Empty:
            pass
        self.root.after(100, self.update_logs)

    # ---------------- Control ----------------
    def test_connection(self):
        repo_url = self.repo_var.get().strip()
        if not repo_url:
            messagebox.showerror("Invalid Input", "Repository URL cannot be empty")
            return
        
        self.test_btn.config(state=tk.DISABLED)
        
        def test_thread():
            success, error = test_connection(repo_url, self.log_queue)
            if success:
                messagebox.showinfo("Success", "Connection to repository successful!")
            else:
                messagebox.showerror("Connection Failed", error)
            self.test_btn.config(state=tk.NORMAL)
        
        threading.Thread(target=test_thread, daemon=True).start()

    def start(self):
        if self.is_running:
            return
        try:
            self.repo_url = self.repo_var.get().strip()
            self.output_dir = self.output_var.get().strip()
            self.start_rev = int(self.start_rev_var.get())
            self.end_rev = int(self.end_rev_var.get())
            self.timeout = int(self.timeout_var.get())
            self.max_retries = int(self.retry_var.get())
            self.max_threads = int(self.threads_var.get())
            
            if not self.repo_url:
                raise ValueError("Repository URL cannot be empty")
            if self.start_rev > self.end_rev:
                raise ValueError("Start revision must be <= End revision")
            if self.timeout < 5:
                raise ValueError("Timeout must be at least 5 seconds")
            if self.max_retries < 1:
                raise ValueError("Max retries must be at least 1")
            if self.max_threads < 1 or self.max_threads > 50:
                raise ValueError("Concurrent threads must be between 1 and 50")
                
            ext_str = self.ext_var.get().strip()
            self.extensions_set = {e.strip().lower() for e in ext_str.split(",") if e.strip()} if ext_str else set()
        except ValueError as e:
            messagebox.showerror("Invalid Input", str(e))
            return

        self.is_running = True
        self.start_btn.config(state=tk.DISABLED)
        self.test_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.results.clear()
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.is_running = False
        self.log("Stopping download...")
        self.stop_btn.config(state=tk.DISABLED)

    # ---------------- Download Logic ----------------
    def run(self):
        # Test connection first
        success, error = test_connection(self.repo_url, self.log_queue)
        if not success:
            self.log(f"Connection test failed: {error}")
            self.log("Aborting download.")
            self.start_btn.config(state=tk.NORMAL)
            self.test_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.is_running = False
            return

        total_revs = self.end_rev - self.start_rev + 1
        completed = 0
        threads = []
        semaphore = threading.Semaphore(self.max_threads)
        
        def fetch_with_semaphore(rev):
            nonlocal completed
            with semaphore:
                if not self.is_running:
                    return
                fetch_commit_log(rev, self.repo_url, self.extensions_set, 
                               self.log_queue, self.results, 
                               self.timeout, self.max_retries)
                completed += 1
                self.progress_var.set(f"Progress: {completed}/{total_revs} revisions")

        for rev in range(self.start_rev, self.end_rev + 1):
            if not self.is_running:
                break
            t = threading.Thread(target=fetch_with_semaphore, args=(rev,))
            t.start()
            threads.append(t)

        # Wait for all threads to finish
        for t in threads:
            t.join()

        if self.results:
            # Sort by revision
            self.results.sort(key=lambda x: x[0])
            repo_name = extract_repo_name(self.repo_url)
            output_file = Path(self.output_dir) / f"{repo_name}_commit_logs.txt"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("revision | filepath | change type | line number | commit message | author | date\n")
                for row in self.results:
                    f.write(" | ".join(str(r) for r in row) + "\n")
            self.log(f"All revisions merged into {output_file}")
            self.log(f"Total entries: {len(self.results)}")

        self.start_btn.config(state=tk.NORMAL)
        self.test_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.is_running = False
        self.progress_var.set("Complete!")
        self.log("All done!")

# ---------------- Main ----------------
def main():
    root = tk.Tk()
    app = SVNCommitLoggerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()