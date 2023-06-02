"""
@author: Tien Nguyen
@date  : 2023-06-01
"""
import datetime

def get_time_now():
    now = datetime.datetime.now()
    return (now.year, now.month, now.day, now.hour, now.minute, now.second)
