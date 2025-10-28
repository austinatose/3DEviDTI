import os
import re
from pathlib import Path

STRUCT_ROOT = Path("structures")

# count number of folders in STRUCT_ROOT
num_folders = sum(1 for item in STRUCT_ROOT.iterdir() if item.is_dir())
print(f"Number of folders in {STRUCT_ROOT}: {num_folders}")

# print the file in each folder
for folder in STRUCT_ROOT.iterdir():
    if folder.is_dir():
        files = list(folder.iterdir())
        print(f"Files in folder {folder.name}: {[file.name for file in files]}")