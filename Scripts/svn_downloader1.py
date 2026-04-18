import subprocess
import os
import shutil
from multiprocessing import Pool
from pathlib import Path
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import queue

# Configuration
DEFAULT_REPO_URL = "svn://svn.code.sf.net/p/ctags/code/trunk"
TEMP_DIR = "./temp_downloads"

# Default extensions
DEFAULT_EXTENSIONS = ".c, .h, .cpp, .hpp, .py, .java, .js, .go, .rs"

# Worker functions (must be at module level for pickling on Windows)
def cleanup_folder(folder_path, keep_extensions):
    removed_count = 0
    kept_count = 0
    
    for root, dirs, files in os.walk(folder_path, topdown=False):
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix.lower() not in keep_extensions:
                try:
                    file_path.unlink()
                    removed_count += 1
                except Exception:
                    pass
            else:
                kept_count += 1
        
        for dir_name in dirs:
            dir_path = Path(root) / dir_name
            try:
                if dir_path.exists() and not any(dir_path.iterdir()):
                    dir_path.rmdir()
            except Exception:
                pass
    
    return kept_count, removed_count

def download_revision_worker(args):
    """Worker function for downloading a single revision."""
    rev, repo_url, output_dir, extensions = args
    
    temp_target = Path(TEMP_DIR) / f"rev_{rev}"
    final_target = Path(output_dir) / f"Revision_{rev}"
    
    try:
        temp_target.mkdir(parents=True, exist_ok=True)
        
        cmd = ["svn", "export", "-r", str(rev), repo_url, 
              str(temp_target), "--force", "--quiet"]
        
        result = subprocess.run(cmd, stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE, text=True)
        
        if result.returncode != 0:
            if temp_target.exists():
                shutil.rmtree(temp_target)
            return {
                'rev': rev,
                'status': 'error',
                'message': result.stderr.strip()
            }
        
        kept, removed = cleanup_folder(temp_target, extensions)
        
        if kept == 0:
            shutil.rmtree(temp_target)
            return {
                'rev': rev,
                'status': 'warning',
                'message': 'No matching files'
            }
        
        final_target.parent.mkdir(parents=True, exist_ok=True)
        if final_target.exists():
            shutil.rmtree(final_target)
        shutil.move(str(temp_target), str(final_target))
        
        return {
            'rev': rev,
            'status': 'success',
            'kept': kept,
            'removed': removed
        }
        
    except Exception as e:
        if temp_target.exists():
            shutil.rmtree(temp_target)
        return {
            'rev': rev,
            'status': 'error',
            'message': str(e)
        }

class SVNDownloaderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SVN Revision Downloader")
        self.root.geometry("900x700")
        self.root.configure(bg="#1e1e2e")
        
        self.is_running = False
        self.log_queue = queue.Queue()
        
        # Configure style
        self.setup_styles()
        
        # Create GUI
        self.create_widgets()
        
        # Start log updater
        self.update_logs()
    
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        bg_color = "#1e1e2e"
        card_color = "#2d2d44"
        accent_color = "#89b4fa"
        text_color = "#cdd6f4"
        success_color = "#a6e3a1"
        error_color = "#f38ba8"
        
        style.configure("Card.TFrame", background=card_color, relief="flat")
        style.configure("Main.TFrame", background=bg_color)
        style.configure("Title.TLabel", background=bg_color, foreground=accent_color, 
                       font=("Segoe UI", 20, "bold"))
        style.configure("Header.TLabel", background=card_color, foreground=text_color, 
                       font=("Segoe UI", 11, "bold"))
        style.configure("Body.TLabel", background=card_color, foreground=text_color, 
                       font=("Segoe UI", 10))
        style.configure("Start.TButton", font=("Segoe UI", 11, "bold"))
        style.configure("Stop.TButton", font=("Segoe UI", 11, "bold"))
        
        # Progress bar style
        style.configure("Custom.Horizontal.TProgressbar", 
                       background=accent_color, troughcolor=card_color, 
                       bordercolor=card_color, lightcolor=accent_color, 
                       darkcolor=accent_color)
    
    def create_widgets(self):
        main_frame = ttk.Frame(self.root, style="Main.TFrame", padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title = ttk.Label(main_frame, text="🚀 SVN Revision Downloader", style="Title.TLabel")
        title.pack(pady=(0, 20))
        
        # Configuration Card
        config_card = ttk.Frame(main_frame, style="Card.TFrame", padding=20)
        config_card.pack(fill=tk.X, pady=(0, 15))
        
        # SVN Repository URL
        ttk.Label(config_card, text="SVN Repository URL:", style="Header.TLabel").grid(
            row=0, column=0, sticky=tk.W, pady=(0, 5))
        self.repo_url_var = tk.StringVar(value=DEFAULT_REPO_URL)
        url_entry = ttk.Entry(config_card, textvariable=self.repo_url_var, 
                             font=("Segoe UI", 10))
        url_entry.grid(row=1, column=0, columnspan=2, sticky=tk.EW, pady=(0, 15))
        
        # Repository Name
        ttk.Label(config_card, text="Output Folder Name:", style="Header.TLabel").grid(
            row=2, column=0, sticky=tk.W, pady=(0, 5))
        self.repo_name_var = tk.StringVar(value="ctags")
        repo_entry = ttk.Entry(config_card, textvariable=self.repo_name_var, 
                              font=("Segoe UI", 10), width=30)
        repo_entry.grid(row=3, column=0, columnspan=2, sticky=tk.EW, pady=(0, 15))
        
        # Start and End Revision
        rev_frame = ttk.Frame(config_card, style="Card.TFrame")
        rev_frame.grid(row=4, column=0, columnspan=2, sticky=tk.EW, pady=(0, 15))
        
        ttk.Label(rev_frame, text="Start Revision:", style="Body.TLabel").pack(
            side=tk.LEFT, padx=(0, 10))
        self.start_rev_var = tk.StringVar(value="1")
        ttk.Entry(rev_frame, textvariable=self.start_rev_var, 
                 font=("Segoe UI", 10), width=10).pack(side=tk.LEFT, padx=(0, 20))
        
        ttk.Label(rev_frame, text="End Revision:", style="Body.TLabel").pack(
            side=tk.LEFT, padx=(0, 10))
        self.end_rev_var = tk.StringVar(value="100")
        ttk.Entry(rev_frame, textvariable=self.end_rev_var, 
                 font=("Segoe UI", 10), width=10).pack(side=tk.LEFT, padx=(0, 20))
        
        ttk.Label(rev_frame, text="Workers:", style="Body.TLabel").pack(
            side=tk.LEFT, padx=(0, 10))
        self.workers_var = tk.StringVar(value="15")
        ttk.Entry(rev_frame, textvariable=self.workers_var, 
                 font=("Segoe UI", 10), width=10).pack(side=tk.LEFT)
        
        # Extensions
        ttk.Label(config_card, text="File Extensions (comma-separated):", 
                 style="Header.TLabel").grid(row=5, column=0, sticky=tk.W, pady=(0, 5))
        self.extensions_var = tk.StringVar(value=DEFAULT_EXTENSIONS)
        ext_entry = ttk.Entry(config_card, textvariable=self.extensions_var, 
                             font=("Segoe UI", 10))
        ext_entry.grid(row=6, column=0, columnspan=2, sticky=tk.EW)
        
        config_card.columnconfigure(0, weight=1)
        
        # Control Buttons
        button_frame = ttk.Frame(main_frame, style="Main.TFrame")
        button_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.start_btn = tk.Button(button_frame, text="▶ Start Download", 
                                   command=self.start_download,
                                   bg="#a6e3a1", fg="#1e1e2e", 
                                   font=("Segoe UI", 12, "bold"),
                                   relief=tk.FLAT, padx=30, pady=12,
                                   cursor="hand2", activebackground="#94d890")
        self.start_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_btn = tk.Button(button_frame, text="■ Stop", 
                                  command=self.stop_download,
                                  bg="#f38ba8", fg="#1e1e2e",
                                  font=("Segoe UI", 12, "bold"),
                                  relief=tk.FLAT, padx=30, pady=12,
                                  cursor="hand2", state=tk.DISABLED,
                                  activebackground="#e17996")
        self.stop_btn.pack(side=tk.LEFT)
        
        # Progress Card
        progress_card = ttk.Frame(main_frame, style="Card.TFrame", padding=20)
        progress_card.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(progress_card, text="Progress", style="Header.TLabel").pack(
            anchor=tk.W, pady=(0, 10))
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_card, variable=self.progress_var,
                                           style="Custom.Horizontal.TProgressbar",
                                           maximum=100, length=400)
        self.progress_bar.pack(fill=tk.X, pady=(0, 10))
        
        self.progress_label = ttk.Label(progress_card, text="Ready to start", 
                                       style="Body.TLabel")
        self.progress_label.pack(anchor=tk.W)
        
        # Log Card
        log_card = ttk.Frame(main_frame, style="Card.TFrame", padding=20)
        log_card.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(log_card, text="📋 Activity Log", style="Header.TLabel").pack(
            anchor=tk.W, pady=(0, 10))
        
        self.log_text = scrolledtext.ScrolledText(log_card, height=15,
                                                  bg="#2d2d44", fg="#cdd6f4",
                                                  font=("Consolas", 9),
                                                  relief=tk.FLAT,
                                                  insertbackground="#cdd6f4")
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Configure log text tags for colors
        self.log_text.tag_config("info", foreground="#89b4fa")
        self.log_text.tag_config("success", foreground="#a6e3a1")
        self.log_text.tag_config("error", foreground="#f38ba8")
        self.log_text.tag_config("warning", foreground="#f9e2af")
    
    def log(self, message, tag="info"):
        self.log_queue.put((message, tag))
    
    def update_logs(self):
        try:
            while True:
                message, tag = self.log_queue.get_nowait()
                self.log_text.insert(tk.END, message + "\n", tag)
                self.log_text.see(tk.END)
        except queue.Empty:
            pass
        self.root.after(100, self.update_logs)
    
    def start_download(self):
        if self.is_running:
            return
        
        try:
            repo_url = self.repo_url_var.get().strip()
            start_rev = int(self.start_rev_var.get())
            end_rev = int(self.end_rev_var.get())
            workers = int(self.workers_var.get())
            repo_name = self.repo_name_var.get().strip()
            extensions_str = self.extensions_var.get()
            
            if not repo_url:
                raise ValueError("Repository URL cannot be empty")
            if not repo_url.startswith(("svn://", "http://", "https://")):
                raise ValueError("Repository URL must start with svn://, http://, or https://")
            if not repo_name:
                raise ValueError("Output folder name cannot be empty")
            if start_rev > end_rev:
                raise ValueError("Start revision must be <= end revision")
            if workers < 1:
                raise ValueError("Workers must be >= 1")
            
            extensions = {ext.strip() for ext in extensions_str.split(',') if ext.strip()}
            if not extensions:
                raise ValueError("At least one extension must be specified")
            
        except ValueError as e:
            messagebox.showerror("Invalid Input", str(e))
            return
        
        self.is_running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.progress_var.set(0)
        
        # Start download in separate thread
        thread = threading.Thread(target=self.run_download, 
                                 args=(repo_url, start_rev, end_rev, workers, repo_name, extensions))
        thread.daemon = True
        thread.start()
    
    def stop_download(self):
        self.is_running = False
        self.log("🛑 Stopping download...", "warning")
        self.stop_btn.config(state=tk.DISABLED)
    
    def run_download(self, repo_url, start_rev, end_rev, workers, repo_name, extensions):
        output_dir = Path(f"./{repo_name}")
        total_revs = end_rev - start_rev + 1
        
        self.log(f"🚀 Starting download from: {repo_url}", "info")
        self.log(f"📊 Revisions: r{start_rev} to r{end_rev}", "info")
        self.log(f"📦 Workers: {workers}", "info")
        self.log(f"📁 Output: {output_dir}", "info")
        self.log(f"🔧 Extensions: {', '.join(sorted(extensions))}", "info")
        self.log("-" * 60, "info")
        
        # Clean up temp directory
        if Path(TEMP_DIR).exists():
            shutil.rmtree(TEMP_DIR)
        Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)
        
        completed = 0
        
        # Prepare arguments for workers
        tasks = [(rev, repo_url, str(output_dir), extensions) 
                 for rev in range(start_rev, end_rev + 1)]
        
        # Process revisions
        try:
            with Pool(processes=workers) as pool:
                for result in pool.imap_unordered(download_revision_worker, tasks):
                    if not self.is_running:
                        pool.terminate()
                        break
                    
                    # Log result
                    rev = result['rev']
                    status = result['status']
                    
                    if status == 'success':
                        kept = result['kept']
                        removed = result['removed']
                        self.log(f"✅ Completed r{rev} (kept {kept}, removed {removed})", "success")
                    elif status == 'error':
                        msg = result['message']
                        self.log(f"❌ Failed r{rev}: {msg}", "error")
                    elif status == 'warning':
                        msg = result['message']
                        self.log(f"⚠️  r{rev}: {msg}", "warning")
                    
                    completed += 1
                    progress = (completed / total_revs) * 100
                    self.progress_var.set(progress)
                    self.progress_label.config(
                        text=f"Completed {completed}/{total_revs} revisions ({progress:.1f}%)")
        except Exception as e:
            self.log(f"❌ Error during download: {e}", "error")
        
        # Cleanup
        if Path(TEMP_DIR).exists():
            shutil.rmtree(TEMP_DIR)
        
        if self.is_running:
            self.log("-" * 60, "info")
            self.log(f"🎉 All done! Saved to {output_dir}", "success")
        else:
            self.log("⚠️  Download stopped by user", "warning")
        
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.is_running = False

def main():
    root = tk.Tk()
    app = SVNDownloaderGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()