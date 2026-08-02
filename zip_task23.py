"""
zip_task23.py — Packaging script for Task 23
"""

import zipfile
import os
import datetime
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def create_zip():
    task_num = 23
    date_str = datetime.date.today().strftime("%Y%m%d")
    zip_name = f"placemux_task{task_num:02d}_{date_str}.zip"

    try:
        with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
            # Add folders
            for folder in ["src", "models", "logs", "data"]:
                if os.path.exists(folder):
                    for root, dirs, files in os.walk(folder):
                        # Exclude pycache
                        if "__pycache__" in root:
                            continue
                        for file in files:
                            file_path = os.path.join(root, file)
                            # Exclude previous large zip files if accidentally in a directory
                            if not file.endswith(".zip"):
                                zf.write(file_path)
            
            # Add specific root files
            for f in [f"run_task{task_num:02d}.bat", "requirements.txt", "README.md", "zip_task23.py"]:
                if os.path.exists(f):
                    zf.write(f)

        logger.info(f"✅ ZIP created successfully: {zip_name}")
    except Exception as e:
        logger.error(f"❌ Failed to create ZIP: {e}", exc_info=True)

if __name__ == "__main__":
    create_zip()
