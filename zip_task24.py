"""
zip_task24.py
Packages Task 24 deliverables into a zip archive per standing instructions.
"""
import zipfile
import os
import datetime
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def main():
    task_num = 24
    date_str = datetime.date.today().strftime("%Y%m%d")
    zip_name = f"placemux_task{task_num:02d}_{date_str}.zip"

    try:
        with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
            for folder in ["src", "models", "logs", "data"]:
                if os.path.exists(folder):
                    for root, dirs, files in os.walk(folder):
                        # Don't zip __pycache__ or .git
                        if "__pycache__" in root or ".git" in root:
                            continue
                        for file in files:
                            if file.endswith(".pyc"):
                                continue
                            file_path = os.path.join(root, file)
                            zf.write(file_path)
            
            for f in [f"run_task{task_num:02d}.bat", "requirements.txt", "README.md", "zip_task24.py"]:
                if os.path.exists(f):
                    zf.write(f)

        logger.info(f"✅ ZIP created successfully: {zip_name}")
    except Exception as e:
        logger.error(f"Failed to create ZIP: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
