import pandas as pd
import os
from glob import glob
import json

pd.set_option('display.max_columns', None)

sources = ['idealista', 'lookup_tables', 'income_opendata', 'opendatabcn-incidents']



def read_data(source: str) -> None:
    local_path = f"data/{source}/"
    for f in os.listdir(local_path):
        print(f)
        path = f"{local_path}{f}"
        try:
            if f.endswith('.json'):
                with open(path, 'r') as j:
                    print(j)
                    contents = json.loads(j.read())
                    df = pd.DataFrame.from_records(contents)    
            elif f.endswith('.csv'):
                    df = pd.read_csv(path)
            else:
                for fname in os.listdir(local_path + '/' + f):
                    if fname.endswith('.parquet'):
                        df = pd.read_parquet(local_path + '/' + f + '/' + fname, engine='pyarrow')
                        # df = pd.concat([df.drop(['detailedType'], axis=1), df['detailedType'].apply(pd.Series)], axis=1)  # splitting a dict value of column into multiple columns
            print(df.head())
        except:
            print(f"Error occured")


for source in sources:
    print("Reading",source,"...")
    read_data(source)