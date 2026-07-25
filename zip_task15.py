"""
zip_task15.py — Package Task 15 deliverables into a versioned ZIP archive.

Standing instructions: always ZIP at end of every task.
"""
import zipfile
import os
import datetime

TASK_NUM = 15
date_str = datetime.date.today().strftime("%Y%m%d")
zip_name = f"placemux_task{TASK_NUM:02d}_{date_str}.zip"

INCLUDE_FOLDERS = ["src", "models", "logs"]
INCLUDE_FILES   = [
    f"run_task{TASK_NUM:02d}.bat",
    "requirements.txt",
    "README.md",
]

with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
    for folder in INCLUDE_FOLDERS:
        if os.path.exists(folder):
            for root, dirs, files in os.walk(folder):
                for file in files:
                    zf.write(os.path.join(root, file))
    for f in INCLUDE_FILES:
        if os.path.exists(f):
            zf.write(f)

print(f"✅ ZIP created: {zip_name}")
