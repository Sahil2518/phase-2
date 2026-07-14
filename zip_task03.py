"""
zip_task03.py — Package Task 03 deliverables into a versioned ZIP archive.

Standing instructions: always create a ZIP at the end of every task.
"""

import zipfile
import os
import datetime
import sys

task_num = 3
date_str = datetime.date.today().strftime("%Y%m%d")
zip_name = f"placemux_task{task_num:02d}_{date_str}.zip"

INCLUDE_DIRS  = ["src", "data", "logs", "models"]
INCLUDE_FILES = [
    f"run_task{task_num:02d}.bat",
    "requirements.txt",
    "README.md",
]

print(f"Creating {zip_name} ...")

try:
    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
        for folder in INCLUDE_DIRS:
            if os.path.exists(folder):
                for root, dirs, files in os.walk(folder):
                    # Skip __pycache__ and .pyc files
                    dirs[:] = [d for d in dirs if d != "__pycache__"]
                    for file in files:
                        if file.endswith(".pyc"):
                            continue
                        full_path = os.path.join(root, file)
                        zf.write(full_path)
                        print(f"  + {full_path}")
            else:
                print(f"  [SKIP] Directory not found: {folder}")

        for fname in INCLUDE_FILES:
            if os.path.exists(fname):
                zf.write(fname)
                print(f"  + {fname}")
            else:
                print(f"  [SKIP] File not found: {fname}")

    size_kb = os.path.getsize(zip_name) / 1024
    print(f"\n[OK] ZIP created: {zip_name}  ({size_kb:.1f} KB)")

except Exception as e:
    print(f"\n[FAIL] ZIP creation failed: {e}")
    sys.exit(1)
