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

def write_csv(
        df, 
        csv_file, 
        sorted=False, 
        by=None
    ) -> None:
    """
    @args:
        - df       : pandas.DataFrame
        - csv_file : str
        - sorted   : boolean
        - by       : list of strings or string
    @desc:
        - Write pandas.DataFrame to csv file
    """
    df = create_df(df)
    if sorted:
        df = df.sort_values(by=by, ascending=False)
    df.to_csv(csv_file, index=False)
