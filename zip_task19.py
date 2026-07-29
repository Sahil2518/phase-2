"""
zip_task19.py — Package Task 19 deliverables into a versioned ZIP archive.
Standing instruction Rule 1: always ZIP at end of every full task.
"""
import zipfile, os, datetime

task_num = 19
date_str = datetime.date.today().strftime("%Y%m%d")
zip_name = f"placemux_task{task_num:02d}_{date_str}.zip"

include_folders = ["src", "models", "logs"]
include_files   = [
    f"run_task{task_num:02d}.bat",
    "requirements.txt",
    "README.md",
]

with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
    for folder in include_folders:
        if os.path.exists(folder):
            for root, dirs, files in os.walk(folder):
                for file in files:
                    # Ignore pycache
                    if "__pycache__" not in root:
                        full = os.path.join(root, file)
                        zf.write(full)
                        print(f"  + {full}")
    for f in include_files:
        if os.path.exists(f):
            zf.write(f)
            print(f"  + {f}")

print(f"\n[OK] ZIP created: {zip_name}")
