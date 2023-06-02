"""
@author: Tien Nguyen
@date  : 2023-05-07
"""
import pickle
import json

from .datetime import get_time_now
from .paths import get_path_basename

def read_txt_file(
        filename: str
    ) -> list:
    with open(filename, 'r') as file:
        data = file.readlines()
    data = [item.strip() for item in data]
    return data

def read_pkl(
        pkl_file: str
    ):
    """
    @desc:
        - reading a pickle file
    """
    with open(pkl_file, 'rb') as file:
        return pickle.load(file)

def write_pkl(
        pkl_file, data
    ) -> None:
    """
    @desc:
        - store data into a pkl file
    """
    with open(pkl_file, 'wb') as file:
        pickle.dump(data, file)

def write_json(
        data_dict: dict, json_file: str
    ) -> None:
    """
    @desc:
        - store data into a json file
    """
    with open(json_file, 'w') as file:
        json.dump(data_dict, file, indent=4)

def read_json(
        json_file: str
    ) -> dict:
    """
    @desc:
        - read json file
    """
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data

def parse_fiel_id(
        image_file: str
    ) -> str:
    basename, _ = get_path_basename(image_file)
    return basename

def create_report_name():
    now = get_time_now()
    report_name = ""
    for item in now[:-1]:
        report_name += str(item)
    report_name += str(now[-1])
    return report_name
