import os 
from pathlib import Path
import logging

file_dir = 'src'

 
file_list = [
    ".github/workflows/.gitkeep",
    f"{file_dir}/__init__.py",
    f"{file_dir}/components/__init__.py",
    f"{file_dir}/utils/__init__.py",
    f"{file_dir}/utils/common.py",
    f"{file_dir}/logging/__init__.py",
    f"{file_dir}/config/__init__.py",
    f"{file_dir}/config/configuration.py",
    f"{file_dir}/pipeline/__init__.py",
    f"{file_dir}/entity/__init__.py",
    f"{file_dir}/constants/__init__.py",
    "config/config.yaml",
    "params.yaml",
    "app.py",
    "main.py",
    "Dockerfile",
    "requirements.txt",
    "research/Capstone.ipynb",
]

try :
    for filepath in file_list:
        filepath = Path(filepath)
        filedir,filename = os.path.split(filepath)
        if filedir != '':
            os.makedirs(filedir,exist_ok=True)
        if not (os.path.exists(filepath) or (os.path.getsize==0)):
            with open(filepath,'w') as f:
                pass
            print(f'File Created in Path {filepath}')
except Exception as e:
        print(e)
    
