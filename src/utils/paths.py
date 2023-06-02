"""
@author : Tien Nguyen
@date   : 2023-04-26
"""
import os
import glob

import constants

def join_path(
        segments: tuple
    ) -> str:
    return os.path.join(*segments)

def split_file_path(
        path: str
    ) -> tuple: 
    return os.path.split(path)

def get_cwd(
    ) -> str:
    return os.getcwd()

def list_files(
        path: tuple
    ) -> list:
    """
    @desc:
        - Find files in path
    """
    return glob.glob(path)

def get_path_basename(
        path: str
    ) -> tuple:
    segments = os.path.basename(path).split('.')
    file_ext = segments[-1]
    basename = '.'.join(segments[:-1])
    return basename, file_ext

def clean_dir(data_dir):
    """
    @desc:
        - Remove all images and annos files in images and annos directory
    """
    os.system(f'find {data_dir} -name "*.pkl" -print0 | xargs -0 rm')
