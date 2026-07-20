"""
zip_task10.py — Creates a zip archive for Friend Circle Task 10 deliverables.
"""

import zipfile, os, datetime

task_num = 10
date_str = datetime.date.today().strftime("%Y%m%d")
zip_name = f"friend_circle_task{task_num:02d}_{date_str}.zip"

with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in ["FriendListTask.kt", "run_task10.bat", "zip_task10.py"]:
        if os.path.exists(f):
            zf.write(f)

print(f"ZIP created: {zip_name}")
