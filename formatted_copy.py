import pandas as pd
import os
import json
import time
import geopandas as gpd
from shapely.geometry import Point
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
from pyspark.sql import functions as F
import pymongo
import logging


sources = ['idealista', 'lookup_tables', 'income_opendata', 'opendatabcn-incidents']
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
start = time.time()
spark = SparkSession.builder.appName("LAB2").getOrCreate()

logging.getLogger().setLevel(logging.INFO)

def read_income_cols(data) -> pd.DataFrame:
    """This functions handles the read data regarding the income from opencatabcn. 
    Its purpose is to convert the json data in a way they are easily converted to dataframes later.
    The data were nested and had lists of dictionairies as values. 
    Thus we explode these values and we have multiplied rows by the years related in our neighbourhood 
    instead of having one observation per neighbourhood, with year pop and rfd in a list of dictionairies
    """
    neighborhood_data = []
    for entry in data:
        neigh_name = entry["neigh_name "]
        district_id = entry["district_id"]
        district_name = entry["district_name"]
        for info in entry["info"]:
            year = info["year"]
            pop = info["pop"]
            RFD = info["RFD"]
            neighborhood_data.append({
                "neigh_name": neigh_name,
                "district_id": district_id,
                "district_name": district_name,
                "year": year,
                "pop": pop,
                "RFD": RFD
            })
    return neighborhood_data



def read_idealista_cols(df) -> pd.DataFrame:
    """This functions handles the read data regarding idealista. 
    More specifically it splits a dict value of two columns into multiple columns, with these columns being the keys and values of the originals, 
    since the values in the original were dictionairies.
    """
    if 'detailedType' in df.columns:
        df = pd.concat([df.drop(['detailedType'], axis=1), df['detailedType'].apply(pd.Series)], axis=1)  # splitting a dict value of column into multiple columns
    if 'suggestedTexts' in df.columns:
        df = pd.concat([df.drop(['suggestedTexts'], axis=1), df['suggestedTexts'].apply(pd.Series)], axis=1)
    return df



def read_data(source: str) -> pd.DataFrame:
    """This functions reads the data that were provided to us along with the third extra dataset. 
    Regarding the source we give as a parameter it behaves differently.
    If the path given contains data json it read them and convert them to pandas dataframe, 
    but if the source was 'income_opendata' it uses the read_income_cols() function to manipulate the columns. 
    If the source contains csv files or parquet files they are converted as well to to pd. Dataframes.
    In all cases, but here it happens in csvs from incidents and idealista parquet files, 
    the multiple files inside those folders are merged into one dataframe.
    For idealista it uses the read_idealista_cols for proper manipulation of some cols.
    """
    #local_path = f"BDM-project-2/data/{source}/"  #path for debugger
    local_path = f"data/{source}/"
    df = []
    for f in os.listdir(local_path):
        path = f"{local_path}{f}"
        try:
            if f.endswith('.json'):
                with open(path, 'r', encoding='utf-8') as j:
                    if source == 'income_opendata':
                        data = json.loads(j.read())
                        income = read_income_cols(data)
                        df.append(pd.DataFrame(income))

                    elif source == 'lookup_tables':
                        contents = json.loads(j.read())
                        df.append(pd.DataFrame.from_records(contents))

            elif f.endswith('.csv') and source=='opendatabcn-incidents':
                    # After 2016 column names change,
                    # make all equal
                    incid = pd.read_csv(path).rename(columns={"Codi Incident": "Codi_Incident",
                                                              "Descripció Incident": "Descripcio_Incident",
                                                              "Codi districte": "Codi_districte",
                                                              "Nom districte": "Nom_districte",
                                                              "Codi barri": "Codi_barri",
                                                              "Nom barri": "Nom_barri",
                                                              "NK Any": "NK_Any",
                                                              "Mes de any": "Mes_any",
                                                              "Nom mes": "Nom_mes",
                                                              "Número d'incidents GUB": "Numero_incidents_GUB"})
                    
                    if pd.api.types.is_string_dtype(incid["NK_Any"]):
                        incid["NK_Any"] = incid["NK_Any"].astype(int)
                    df.append(incid)

            else:
                for fname in os.listdir(local_path + '/' + f):
                    if fname.endswith('.parquet'):
                        df_loop = pd.read_parquet(local_path + '/' + f + '/' + fname, engine='pyarrow')
                        df_loop["name"] = f
                        df_loop["date"] = pd.to_datetime(f"{f[0:10]}", format="%Y_%m_%d")
                        df_loop["month"] = df_loop["date"].dt.month
                        df_loop["year"] = df_loop["date"].dt.year
                        df.append(df_loop)

        except:
            logging.critical(f"Error ocurred while reading {f} data.")
    df = pd.concat(df, ignore_index=True)
    if source == 'idealista':
        df = read_idealista_cols(df)

    return df


def format(df) -> DataFrame:
    """This functions formats the data and transforms them to spark dataframes. 
    It converts object columns to string, it checks and deletes fully duplicated rows, 
    checks for missing values and drops columns with more than 80% missing 
    and if the dataframe contains '_id' columns it checks for duplicated id, and if True it removes the duplicated row.
    This functions can handle any possible dataframe given, so it is for automated use.
    """
    for col in df.select_dtypes(include='object'):
        df[col] = df[col].astype(str)
    # df = df[~df.astype(str).duplicated()]
    df_sp = spark.createDataFrame(df)
    #print((df_sp.count(), len(df_sp.columns)))
    # print(df_sp.show(10,0))
    df_no_dupl = df_sp.drop_duplicates()


    # str_column_list = [item[0] for item in df_no_dupl.dtypes if item[1].startswith('string')]
    nof_rows = df_no_dupl.count()
    cols_drop = []
    for c in df_no_dupl.columns:
        count_na= df_no_dupl.where((df_no_dupl[c].isNull()) | (df_no_dupl[c] == 'None') | (df_no_dupl[c] == 'nan')).count()  # when converting to spark df, the NaN/None values are becoming string values.
        if count_na/nof_rows > 0.8:
            cols_drop.append(c)
    df_no_miss = df_no_dupl.drop(*cols_drop)
    # print((df_no_miss.count(), len(df_no_miss.columns)))

    if '_id' in df_no_miss.columns:
        counter_all = df_no_miss.count()
        counter_dupl = df_no_miss.drop_duplicates(subset=['_id']).count()
        if counter_all > counter_dupl:
            return df_no_miss.drop_duplicates(subset=['_id'])           

    return df_no_miss
    # return df_sp


def _infer_neighborhood(df) -> DataFrame:
    """ Infers the neighborhood based on the latitude and longitude and returns the BCN Open Data nomenclature.
     In case its outside the neighborhood coordinates limit, obtains the open data bcn using the lookup table."""

    # Read neighborhood coordinates delimitation
    neig_coord = ( gpd.read_file("data/BCN_NEIGHBORHOODS/NEIGHBORHOODS_DELIMITATIONS.json")
                        .query("TIPUS_UA == 'BARRI'").to_crs(crs=4326) )
    
    # Assign new column with imputed neighborhood using pandas structure
    pandas_df = df.toPandas()
    pandas_df["neighborhood"] = pandas_df.apply(lambda x: _identify_neighborhood(x, neig_coord), axis=1)
    return spark.createDataFrame(pandas_df)


def _identify_neighborhood(df, neig_coord) -> str:
    """ Assigns the neighboor that contains the given coordinate,
    in the case it is outside the delimitation of the neighborhood,
    assigns the nearest neighborhood. """
    return neig_coord.iloc[neig_coord.sindex.nearest(Point(df["longitude"], df["latitude"]))[1][0]]["NOM"]


def _select_columns(df, keep=[], drop=[]) -> DataFrame:
    """Selects the desired columns of the dataframe"""
    if keep:
        return df[keep]
    
    elif drop:
        return df.drop(*drop)
    
    return df


def store_to_mongo(client, source, collection_name, df):
        db = client[source]
        collection = db[collection_name]
        results = df.toJSON().map(lambda j: json.loads(j)).collect()
        for res in results:
            collection.insert_one(res)



if __name__ == "__main__":

    logging.info("Starting Extraction and Formatting of data from Landing zone.")

    logging.info("Extracting idealista data.")
    df_id = read_data('idealista')
    logging.info("Formatting idealista data.")
    df_id = format(df_id)
    df_id = _select_columns(df_id, drop=["date", "month"])

    logging.info("Extracting income data.")
    df_in = read_data("income_opendata")
    df_in["year"] += 5 # Patch to make tables join (Transform years from [2007-2017] to [2012-2022])
    logging.info("Formatting income data.")
    df_in = format(df_in)
    df_in = _select_columns(df_in, keep=["neigh_name", "year", "pop", "RFD"])

    logging.info("Extracting incidents data.")
    df_incid = read_data('opendatabcn-incidents')
    logging.info("Formatting incidents data.")
    df_incid = format(df_incid)
    df_incid = _select_columns(df_incid, keep=["Codi_Incident", "Descripcio_Incident", "Nom_barri",
                                               "NK_Any", "Numero_incidents_GUB"])
    df_incid = df_incid.groupby(["Codi_Incident", "Descripcio_Incident", "Nom_barri", "NK_Any"]).agg(F.sum("Numero_incidents_GUB").alias("Numero_incidents_GUB"))

    logging.info("Extracting lookup tables.")
    df_l = read_data('lookup_tables')
    logging.info("Formatting lookup tables.")
    df_l = format(df_l)

    logging.info("Finished reading and formatting data.")


    logging.info("Starting the reconciliation.")
    logging.info("Imputing NaN idealista neighborhood using coordinates while using the BCN-OpenData Nomenclature.")
    df_id = _infer_neighborhood(df_id)
    logging.info("Finished reconciliation.")

    logging.info("Starting uploading data to formatted zone.")
    try:
        client_mongo = pymongo.MongoClient("mongodb+srv://alextrem:1234@cluster0.x1yh6no.mongodb.net/?retryWrites=true&w=majority")
        
        logging.info("Starting uploading idealista data.")
        store_to_mongo(client_mongo,'all', "idealista", df_id)
        logging.info("Successfully uploaded idealista data.")
        time.sleep(2)

        logging.info("Starting uploading lookup tables.")
        store_to_mongo(client_mongo,'all', "lookup_tables", df_l)
        logging.info("Successfully uploaded lookup tables.")
        time.sleep(2)

        logging.info("Starting uploading income data.")
        store_to_mongo(client_mongo,'all', "income", df_in)
        logging.info("Successfully uploaded income data.")
        time.sleep(2)

        logging.info("Starting uploading incidents data.")
        store_to_mongo(client_mongo,'all', "incidents", df_incid)
        logging.info("Successfully uploaded incidents data.")
        time.sleep(2)
        client_mongo.close()

        logging.info("All data has been successfuly uploaded to formatted zone.")

        exec_time = time.time() - start
        logging.info(f"Full execution time of {exec_time} seconds.")

    except ConnectionResetError as e:
        logging.critical(f"ConnectionResetError ocurred: {e}.")
    except Exception as e:
        logging.critical(f"An unexpected error occurred: {e}.")
