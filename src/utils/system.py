"""
@author: Tien Nguyen
@date  : 2023-05-15
"""

import os

def exec_system_call(
        command: str
    ):
    """
    @desc:
        - Call system command
    """
    os.system(command)

def list_subdirs(
        path: str
    ):
    return os.listdir(path)

def clean_dir(data_dir):
    """
    @desc:
        - Remove all images and annos files in images and annos directory
    """
    os.system(f'find {data_dir} -name "*.pkl" -print0 | xargs -0 rm')

def mkdir(
        path: str
    ) -> None:
    os.makedirs(path, exist_ok=True)
