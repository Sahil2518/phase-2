import zipfile
import os
import datetime

def main():
    task_num = 21
    date_str = datetime.date.today().strftime("%Y%m%d")
    zip_name = f"placemux_task{task_num:02d}_{date_str}.zip"
    
    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
        # Include specific folders
        for folder in ["src", "models", "logs", "data"]:
            if os.path.exists(folder):
                for root, dirs, files in os.walk(folder):
                    for file in files:
                        # Exclude pycache and large unneeded files if any
                        if "__pycache__" not in root:
                            zf.write(os.path.join(root, file))
                            
        # Include root files
        for f in [f"run_task{task_num:02d}.bat", "requirements.txt", "README.md", f"zip_task{task_num:02d}.py"]:
            if os.path.exists(f):
                zf.write(f)

    print(f"✅ ZIP created: {zip_name}")

if __name__ == "__main__":
    main()
