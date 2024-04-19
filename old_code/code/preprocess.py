import asyncio
import os

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

# Adapted from the API documentation
idealista_dtypes = {
    "address": "string",
    "bathrooms": "int",
    "country": "string",
    "distance": "string",
    "district": "string",
    "exterior": "boolean",
    "floor": "string",
    "hasVideo": "boolean",
    "latitude": "float",
    "longitude": "float",
    "municipality": "string",
    "neighborhood": "string",
    "numPhotos": "int",
    "operation": "string",
    "price": "int",
    "propertyCode": "int",
    "province": "string",
    "region": "string",
    "rooms": "int",
    "showAddress": "boolean",
    "size": "int",
    "subregion": "string",
    "thumbnail": "string",
    "url": "string",
    "status": "string",
    "newDevelopment": "boolean",
    "tenantGender": "string",
    "garageType": "string",
    "parkingSpace": "object",
    "hasLift": "boolean",
    "newDevelopmentFinished": "boolean",
    "isSmokingAllowed": "boolean",
    "priceByArea": "float",
    "detailedType": "object",
    "externalReference": "string",
}

income_dtypes = {
    "Any": "int",
    "Codi_Districte": "int",
    "Nom_Districte": "string",
    "Codi_Barri": "int",
    "Nom_Barri": "string",
    "Població": "int",
    "Índex RFD Barcelona = 100": "float",
}

incident_dtypes = {
    "Codi_Incident": "string",
    "Descripcio_Incident": "string",
    "Codi_districte": "int",
    "Nom_districte": "string",
    "Codi_barri": "int",
    "Nom_barri": "string",
    "NK_Any": "int",
    "Mes_any": "int",
    "Nom_mes": "string",
    "Numero_incidents_GUB": "int",
}

incident_column_old2new = {
    "Codi Incident": "Codi_Incident",
    "Descripció Incident": "Descripcio_Incident",
    "Codi districte": "Codi_districte",
    "Nom districte": "Nom_districte",
    "Codi barri": "Codi_barri",
    "Nom barri": "Nom_barri",
    "NK Any": "NK_Any",
    "Mes de any": "Mes_any",
    "Nom mes": "Nom_mes",
    "Número d'incidents GUB": "Numero_incidents_GUB",
}


def _check_data(df: pd.DataFrame, name: str, primary_key: list) -> None:
    """ Check if primary keys are unique/contain nans and provides a general summary of the data. 
    :param df: Dataframe containing the data to check.
    :param primary_key: Primary keys of the data.
    """

    # Check no NaN in primary_keys
    assert not df[primary_key].isna().sum().any(), "NaNs present in Primary Keys"

    # Check duplicated primary_keys
    num_duplicated = df[primary_key].duplicated().sum()
    print(f"There are {num_duplicated} observations with repeated primary keys")

    if num_duplicated > 0:
        print("Proceding to keep only one observation per primary key")
        df = df.drop_duplicates(subset=primary_key)

    # Print general statistics of the dataframe
    print(f"----General Statistics of the {name} data----")
    print(f"Contains {len(df)} rows")
    print(f"Contains {len(df.columns)} columns")

    # Print how many columns present NaNs and which percentage
    nans = df.isna().sum()
    print(f"There are {len(nans[nans>0])} columns presenting NaNs and their percentage is:")
    print(f"{round(100*nans[nans>0]/len(df), 1)}")


def _read_housing() -> pd.DataFrame:
    """ Reads idealista data. """
    housing_dfs = []
    for filename in os.listdir("data/idealista"):
        f = pd.read_json(os.path.join("data/idealista", filename), dtype=idealista_dtypes)
        # Ignore empty dataframes
        if f.shape[0] == 0:
            continue
        f["date"] = pd.to_datetime(f"{filename[0:10]}", format="%Y_%m_%d")
        f["month"] = f["date"].dt.month
        f["year"] = f["date"].dt.year
        housing_dfs.append(f)
    housing = pd.concat(housing_dfs)

    # Return only housing in Barcelona
    return housing[housing["municipality"] == "Barcelona"]


def _read_income() -> pd.DataFrame:
    """ Reads income data. """
    income_dfs = []
    for filename in os.listdir("data/opendatabcn-income"):
        f = pd.read_csv(os.path.join("data/opendatabcn-income", filename))
        # Ignore empty dataframes
        if f.shape[0] == 0:
            continue
        # Handle invalid RFD for special district 99 (unassigned)
        f["Índex RFD Barcelona = 100"] = pd.to_numeric(f["Índex RFD Barcelona = 100"], errors="coerce")
        # Set types
        f = f.astype(income_dtypes)
        # Rename columns into English
        f = f.rename(columns={"Any": "year",
                              "Nom_Barri": "neighborhood",
                              "Població": "population",
                              "Índex RFD Barcelona = 100": "yearly_index_RFD_100"
        })
        income_dfs.append(f)
    income = pd.concat(income_dfs)
    return income.drop(columns={"Codi_Districte",
                                "Nom_Districte",
                                "Codi_Barri"})


def _read_incidents() -> pd.DataFrame:
    """ Reads income data. """
    incidents_dfs = []
    for filename in os.listdir("data/opendatabcn-incidents"):
        f = pd.read_csv(os.path.join("data/opendatabcn-incidents", filename))
        # Ignore empty dataframes
        if f.shape[0] == 0:
            continue
        # Handle change of column names 2015 -> 2016
        f = f.rename(columns=incident_column_old2new)
        # Set types
        f = f.astype(incident_dtypes)
        # Rename columns into English
        f = f.rename(columns={"Codi_Incident": "incident_code",
                              "Descripcio_Incident": "incident_description",
                              "Nom_barri": "neighborhood",
                              "NK_Any": "year",
                              "Mes_any": "month",
                              "Numero_incidents_GUB": "monthly_n_incidents"
        })
        # Remove the leading and trailing space characters
        f["incident_description"] = f["incident_description"].str.strip()
        incidents_dfs.append(f)
    incidents = pd.concat(incidents_dfs)

    return incidents.drop(columns={"Codi_districte",
                                   "Nom_districte",
                                   "Codi_barri",
                                   "Nom_mes"})


def _infer_neighborhood(df) -> pd.DataFrame:
    """ Infers the neighborhood based on the latitude and longitude and returns the BCN Open Data nomenclature.
     In case its outside the neighborhood coordinates limit, obtains the open data bcn using the lookup table."""

    # Read neighborhood coordinates delimitation
    neig_coord = ( gpd.read_file("data/BCN_UNITATS_ADM/UNITATS_ADM_POLIGONS.json")
                        .query("TIPUS_UA == 'BARRI'").to_crs(crs=4326) )
    
    df["neighborhood"] = df.apply(lambda x: _identify_neighborhood(x, neig_coord), axis=1)
    return df


def _identify_neighborhood(df, neig_coord) -> str:
    """ Assigns the neighboor that contains the given coordinate,
    in the case it is outside the delimitation of the neighborhood,
    assigns the nearest neighborhood. """
    return neig_coord.iloc[neig_coord.sindex.nearest(Point(df["longitude"], df["latitude"]))[1][0]]["NOM"]


async def _task_1_idealista() -> pd.DataFrame:
    """ Reads, check and transforms the idealista dataset. """

    housing = ( _read_housing()
                .pipe(_infer_neighborhood) )
    _check_data(housing, "housing", primary_key = ["propertyCode", "date"])
    return housing


async def _task_2_income() -> pd.DataFrame:
    """ Reads, check and transforms the income dataset. """

    income = _read_income()
    _check_data(income, "income", primary_key=["year", "neighborhood"])
    return income


async def _task_3_incidents() -> pd.DataFrame:
    """ Reads, check and transforms the incidents dataset. """
    incidents = _read_incidents()
    _check_data(incidents, "incidents", primary_key=["incident_code", "neighborhood", "year", "month"])
    return incidents


async def preprocess() -> pd.DataFrame:
    """ Merges the three data sources of housing, income and incidents. """

    # Asynchronous tasks
    tasks = [
        asyncio.create_task(_task_1_idealista()),
        asyncio.create_task(_task_2_income()),
        asyncio.create_task(_task_3_incidents())
    ]

    housing, income, incidents = await asyncio.gather(*tasks)

    # Merge housing with incidents
    housing_incidents = housing.merge(incidents, on=["neighborhood", "year", "month"], how="outer")

    # Merge with income
    total = housing_incidents.merge(income, on=["neighborhood", "year"], how="outer")

    return total


if __name__ == "__main__":
    asyncio.run(preprocess())
