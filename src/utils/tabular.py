"""
@author : Tien Nguyen
@date   : 2023-06-02
"""
import pandas

def create_df(
        df: dict
    ) -> pandas.DataFrame:
    """
    @desc:
        - Create pandas.DataFrame from dict
    """
    return pandas.DataFrame(df)
