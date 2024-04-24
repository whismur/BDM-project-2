import pandas as pd
import asyncio
import os
import json

sources = ['idealista', 'lookup_tables', 'income_opendata', 'opendatabcn-incidents']

def read_data(source: str) -> pd.DataFrame:
    local_path = f"data/{source}/"
    df = []
    for f in os.listdir(local_path):
        print(f)
        path = f"{local_path}{f}"
        try:
            if f.endswith('.json') and source.isin(['lookup_tables', 'income_opendata']):
                with open(path, 'r') as j:
                    print(j)
                    contents = json.loads(j.read())
                    return pd.DataFrame.from_records(contents)
            elif f.endswith('.csv') and source == 'opendatabcn-incidents':
                    return pd.read_csv(path)
            else:
                
                for fname in os.listdir(local_path + '/' + f):
                    if fname.endswith('.parquet'):
                        df_loop = pd.read_parquet(local_path + '/' + f + '/' + fname, engine='pyarrow')
                        df_loop["name"] = f
                        df.append(df_loop)
                        # df = pd.concat([df.drop(['detailedType'], axis=1), df['detailedType'].apply(pd.Series)], axis=1)  # splitting a dict value of column into multiple columns
        except:
            print(f"Error occured")
    
    df = pd.concat(df, ignore_index=True)
    return df



async def _task_1_idealista(source: str) -> pd.DataFrame:
    """ Reads, check and transforms the idealista dataset. """

    housing = ( read_data(source)
               .pipe(_format_housing)
               .pipe(_check_data)
               .pipe(_infer_neighborhood) )

    return housing


async def _task_2_income(source: str) -> pd.DataFrame:
    """ Reads, check and transforms the income dataset. """

    income = ( read_data(source)
             .pipe(_format_housing)
             .pipe(_check_data) )
    
    return income


async def _task_3_incidents(source: str) -> pd.DataFrame:
    """ Reads, check and transforms the incidents dataset. """
    incidents = ( read_data(source)
                .pipe(_format_incidents)
                .pipe(_check_data) )
    
    return incidents


async def formatted() -> pd.DataFrame:
    """ Merges the three data sources of housing, income and incidents. """

    # Asynchronous tasks
    tasks = [
        asyncio.create_task(_task_1_idealista("idealista")),
        asyncio.create_task(_task_2_income("income_opendata")),
        asyncio.create_task(_task_3_incidents("opendatabcn-incidents"))
    ]

    housing, income, incidents = await asyncio.gather(*tasks)

    # Merge housing with incidents
    housing_incidents = housing.merge(incidents, on=["neighborhood", "year", "month"], how="outer")

    # Merge with income
    total = housing_incidents.merge(income, on=["neighborhood", "year"], how="outer")

    return total

if __name__ == "__main__":
    asyncio.run(formatted())
