"""
zip_task14.py - Creates a zip archive for Task 14 deliverables.
"""
import zipfile, os, datetime

task_num = 14
date_str = datetime.date.today().strftime("%Y%m%d")
zip_name = f"placemux_task{task_num:02d}_{date_str}.zip"

include_files = [f"run_task{task_num:02d}.bat", "requirements.txt"]
include_dirs  = ["src", "models", "logs"]

with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
    for folder in include_dirs:
        for root, dirs, files in os.walk(folder):
            for file in files:
                zf.write(os.path.join(root, file))
    for f in include_files:
        if os.path.exists(f):
            zf.write(f)

print(f"ZIP created: {zip_name}")
