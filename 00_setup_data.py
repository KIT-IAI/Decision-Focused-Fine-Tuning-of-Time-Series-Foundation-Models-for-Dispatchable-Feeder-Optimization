import copy
import os
import urllib.request
import zipfile

import numpy as np
import pandas as pd






# Function to download and unzip files
def download_and_unzip(url, directory):
    """_summary_ : This method downloads and unzips the file from the given URL and saves it to the given directory.

    Args:
        url (_type_): _description_
        directory (_type_): _description_
    """
    # Extract the filename from the URL
    filename = os.path.join(directory, os.path.basename(url.split("?")[0]))

    # Download the file
    urllib.request.urlretrieve(url, filename)

    # Unzip the file
    with zipfile.ZipFile(filename, "r") as zip_ref:
        zip_ref.extractall(directory)


# method to download the ausgrid solar home dataset
def download_ausgrid_solar_home_dataset():
    """_summary_ : Downloads the ausgrid Dataset"""

    # check if the data folder exists
    if not os.path.exists("data"):
        os.mkdir("data")
    # create the data folder for ausgrid solar home dataset

    if not os.path.exists("data/ausgrid_solar_home_dataset"):
        os.mkdir("data/ausgrid_solar_home_dataset")
        directory = "data/ausgrid_solar_home_dataset/"
        urls = [
            "https://cdn.ausgrid.com.au/-/media/Documents/Data-to-share/Solar-home-electricity-data/Solar-home-half-hour-data---1-July-2012-to-30-June-2013.zip?rev=de594e37789744738fe747c37e1e67bf",
            "https://cdn.ausgrid.com.au/-/media/Documents/Data-to-share/Solar-home-electricity-data/Solar-home-half-hour-data---1-July-2011-to-30-June-2012.zip?rev=938d7e42fe0f43969fc4144341dacfac",
            "https://cdn.ausgrid.com.au/-/media/Documents/Data-to-share/Solar-home-electricity-data/Solar-home-half-hour-data---1-July-2010-to-30-June-2011.zip?rev=3ba8aee669294858a27cda3f2214aba5",
        ]

        # Loop through each URL and download/unzip the file
        for url in urls:
            download_and_unzip(url, directory)


def parse_date_and_time_and_get_series(df):
    """_summary_ : This method parses the date and time and returns a series."""
    data_and_cols = df[df.columns[4:-1]]
    touples = []
    for index, row in data_and_cols.iterrows():
        initial_date = pd.Timestamp(row.iloc[0])
        for row_entry in range(1, len(row)):
            tmp_tuple = (
                initial_date + row_entry * pd.Timedelta("30min"),
                row.iloc[row_entry],
            )
            touples.append(tmp_tuple)

    if len(touples) > 0:
        idx, values = zip(*touples)
    else:
        idx, values = [], []

    return pd.Series(values, idx)


def generate_factored_version_of_ausgrid_dataset(consumption_df, production_df,
                                                 consumption_factor,
                                                 production_factor, suffix):
    results_df = consumption_df * consumption_factor - production_df * production_factor
    print(
        f"Generated facotred version of the ausgrid dataset with the suffix {suffix}"
    )
    results_df.to_csv(
        f"data/ausgrid_solar_home_dataset/ausgrid_prosumption_{suffix}.csv")


def modify_ausgrid_dataset():
    """_summary_ : This method reads in the ausgrid solar home dataset and modifies it to a single csv file."""

    # checks if the data files already exist
    if (os.path.exists(
            "data/ausgrid_solar_home_dataset/ausgrid_gc_customers.csv")
            and os.path.exists(
                "data/ausgrid_solar_home_dataset/ausgrid_gg_customers.csv")
            and os.path.exists(
                "data/ausgrid_solar_home_dataset/ausgrid_cl_customers.csv")
            and os.path.exists(
                "data/ausgrid_solar_home_dataset/ausgrid_prosumption.csv")
            and os.path.exists(
                "data/ausgrid_solar_home_dataset/ausgrid_prosumption_ldiv2.csv"
            ) and os.path.exists(
                "data/ausgrid_solar_home_dataset/ausgrid_prosumption_ldiv5.csv"
            ) and os.path.exists(
                "data/ausgrid_solar_home_dataset/ausgrid_prosumption_load5.csv"
            ) and os.path.exists(
                "data/ausgrid_solar_home_dataset/ausgrid_prosumption_load2.csv"
            ) and os.path.exists(
                "data/ausgrid_solar_home_dataset/ausgrid_prosumption_factor5.csv"
            ) and os.path.exists(
                "data/ausgrid_solar_home_dataset/ausgrid_prosumption_factor10.csv"
            )):
        return

    # read in all 3 ausgrid csv files
    ausgrid_2010_2011 = pd.read_csv(
        "data/ausgrid_solar_home_dataset/2010-2011 Solar home electricity data.csv",
        low_memory=False,
        skiprows=1,
        parse_dates=True,
    )
    ausgrid_2011_2012 = pd.read_csv(
        "data/ausgrid_solar_home_dataset/2011-2012 Solar home electricity data v2.csv",
        low_memory=False,
        skiprows=1,
        parse_dates=True,
    )
    ausgrid_2012_2013 = pd.read_csv(
        "data/ausgrid_solar_home_dataset/2012-2013 Solar home electricity data v2.csv",
        low_memory=False,
        skiprows=1,
        parse_dates=True,
    )

    # get the date collumn in uniformae date format
    ausgrid_2010_2011["date"] = pd.to_datetime(ausgrid_2010_2011["date"],
                                               format="mixed")
    ausgrid_2011_2012["date"] = pd.to_datetime(ausgrid_2011_2012["date"],
                                               format="%d/%m/%Y")
    ausgrid_2012_2013["date"] = pd.to_datetime(ausgrid_2012_2013["date"],
                                               format="%d/%m/%Y")

    # concatenate the 3 ausgrid datasets
    ausgrid = pd.concat(
        [ausgrid_2010_2011, ausgrid_2011_2012, ausgrid_2012_2013])

    # filter Consumption Category after GC, GG and CL and safe them seperatly
    ausgrid_gc = ausgrid[ausgrid["Consumption Category"] == "GC"]
    ausgrid_gg = ausgrid[ausgrid["Consumption Category"] == "GG"]
    ausgrid_cl = ausgrid[ausgrid["Consumption Category"] == "CL"]

    # group them by customer id

    # generate a time index from the lowest to the highest date 30 minutes resolution
    time_index = pd.date_range(
        start=pd.Timestamp(ausgrid["date"].min()) + pd.Timedelta("30T"),
        end=pd.Timestamp(ausgrid["date"].max()) + pd.Timedelta("1D"),
        freq="30T",
    )  # TODO: Change T into Min

    # generate a datframe with the time index as index and the customer id as columns
    ausgrid_gc_customers = pd.DataFrame(index=time_index,
                                        columns=ausgrid["Customer"].unique())
    ausgrid_gg_customers = pd.DataFrame(index=time_index,
                                        columns=ausgrid["Customer"].unique())
    ausgrid_cl_customers = pd.DataFrame(index=time_index,
                                        columns=ausgrid["Customer"].unique())

    for customer_id in ausgrid["Customer"].unique():
        # get the subdataframe
        sub_df_gc = ausgrid_gc[ausgrid_gc["Customer"] == customer_id]
        sub_df_gg = ausgrid_gg[ausgrid_gg["Customer"] == customer_id]
        sub_df_cl = ausgrid_cl[ausgrid_cl["Customer"] == customer_id]

        # add the series to the dataframe
        tmp_series_gc = parse_date_and_time_and_get_series(sub_df_gc)
        if len(tmp_series_gc) != 0:
            ausgrid_gc_customers[customer_id] = tmp_series_gc
        else:
            ausgrid_gc_customers[customer_id] = 0

        tmp_series_gg = parse_date_and_time_and_get_series(sub_df_gg)
        if len(tmp_series_gg) != 0:
            ausgrid_gg_customers[customer_id] = tmp_series_gg
        else:
            ausgrid_gg_customers[customer_id] = 0

        tmp_series_cl = parse_date_and_time_and_get_series(sub_df_cl)
        if len(tmp_series_cl) != 0:
            ausgrid_cl_customers[customer_id] = tmp_series_cl
        else:
            ausgrid_cl_customers[customer_id] = 0

    # save the dataframes to csv files
    ausgrid_gc_customers.to_csv(
        "data/ausgrid_solar_home_dataset/ausgrid_gc_customers.csv")
    ausgrid_gg_customers.to_csv(
        "data/ausgrid_solar_home_dataset/ausgrid_gg_customers.csv")
    ausgrid_cl_customers.to_csv(
        "data/ausgrid_solar_home_dataset/ausgrid_cl_customers.csv")

    # fill every gap in the data with zeroes
    ausgrid_cl_customers = ausgrid_cl_customers.fillna(0)
    ausgrid_gg_customers = ausgrid_gg_customers.fillna(0)
    ausgrid_gc_customers = ausgrid_gc_customers.fillna(0)

    # calculate prosumption
    # df_final = df_gc + df_cl - df_gg
    ausgrid_prosumption = ausgrid_gc_customers + ausgrid_cl_customers - ausgrid_gg_customers

    ausgrid_consumption = ausgrid_gc_customers + ausgrid_cl_customers
    ausgrid_production = ausgrid_gg_customers

    # save the prosumption to a csv file
    ausgrid_prosumption.to_csv(
        "data/ausgrid_solar_home_dataset/ausgrid_prosumption.csv")

    # generate a factored version of the prosumption
    # load div2
    generate_factored_version_of_ausgrid_dataset(ausgrid_consumption,
                                                 ausgrid_production, 0.5, 1,
                                                 "ldiv2")
    # load div5
    generate_factored_version_of_ausgrid_dataset(ausgrid_consumption,
                                                 ausgrid_production, 0.2, 1,
                                                 "ldiv5")
    # load 5
    generate_factored_version_of_ausgrid_dataset(ausgrid_consumption,
                                                 ausgrid_production, 5, 1,
                                                 "load5")
    # load 2
    generate_factored_version_of_ausgrid_dataset(ausgrid_consumption,
                                                 ausgrid_production, 2, 1,
                                                 "load2")
    # pv factor5
    generate_factored_version_of_ausgrid_dataset(ausgrid_consumption,
                                                 ausgrid_production, 1, 5,
                                                 "factor5")
    # pv factor10
    generate_factored_version_of_ausgrid_dataset(ausgrid_consumption,
                                                 ausgrid_production, 1, 10,
                                                 "factor10")

def generate_daily_gt(id):
    """
    This method generates the daily ground truth for a given building id.
    """
    data_ausgrid_building = pd.read_csv("data/ausgrid_solar_home_dataset/ausgrid_prosumption.csv", parse_dates=["time"], index_col="time").resample("1h", closed='right').sum()[str(id)]

    #display(data_ausgrid_building)
  
    list_of_data = []
    for i in range(0, 24):
        list_of_data.append(data_ausgrid_building.shift(-i))

    data_ausgrid_building = pd.concat(list_of_data, axis=1)

    data_ausgrid_building.columns = [str(i) for i in range(0, 24)]
    
    data_ausgrid_building_daily_test = data_ausgrid_building.between_time("00:00:00", "00:00:00")[[str(i) for i in range(0, 24)]].loc["2012-07-08":"2013-06-30"].dropna()

    # save data
    data_ausgrid_building_daily_test.to_csv(f"data/ausgrid_solar_home_dataset/daily/gt_daily_{id}.csv")






# main method for setup
def setup():
    """_summary_ : This method sets up the project by downloading the data and creating the necessary folders."""
    # create a folder named data if it does not exist
    if not os.path.exists("data"):
        os.mkdir("data")

    # Download the Ausgrid Solar Home Dataset
    download_ausgrid_solar_home_dataset()
    modify_ausgrid_dataset()

    # creates daily gt folder
    os.makedirs("data/ausgrid_solar_home_dataset/daily", exist_ok=True)
    for i in range(101, 301):
        generate_daily_gt(i)
    

    print("Setup done!")

if __name__ == "__main__":
    setup()



