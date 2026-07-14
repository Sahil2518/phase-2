"""
zip_task04.py — Archive Task 04 deliverables.

Creates: placemux_task04_YYYYMMDD.zip
"""

import zipfile
import os
import datetime

task_num = 4
date_str = datetime.date.today().strftime("%Y%m%d")
zip_name = f"placemux_task{task_num:02d}_{date_str}.zip"

# Folders to include
FOLDERS = ["src", "models", "logs", "data"]

# Top-level files to include
TOP_FILES = [
    f"run_task{task_num:02d}.bat",
    "requirements.txt",
]

with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
    for folder in FOLDERS:
        if not os.path.isdir(folder):
            continue
        for root, dirs, files in os.walk(folder):
            # Skip __pycache__
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for file in files:
                filepath = os.path.join(root, file)
                zf.write(filepath)
                print(f"  + {filepath}")

    for f in TOP_FILES:
        if os.path.exists(f):
            zf.write(f)
            print(f"  + {f}")

print(f"\n{'='*50}")
print(f"  ZIP created: {zip_name}")
print(f"{'='*50}")
