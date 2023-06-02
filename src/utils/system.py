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
