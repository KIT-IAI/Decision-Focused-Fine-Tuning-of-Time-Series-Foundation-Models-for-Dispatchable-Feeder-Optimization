import numpy as np
import os
import pandas as pd
from src.utils import get_data_test

from src.configuration import MORAI_CONFIG, DATATSET_BOUNDS

def get_shifted_prediction(building_test_set, shift):
    indexes = []
    values = []
    for i in range(len(building_test_set)):
        index = building_test_set.get_time_by_index(i)
        value = building_test_set.get_naive_shift(i, shift, 42)
        indexes.append(index)
        values.append(value)
    pred_df = pd.DataFrame(values, index=indexes, columns=[f"{i}" for i in range(0, 42)])

    pred_df.index.name = "time"

    return pred_df

if __name__ == "__main__":
    #
    # Parse the arguments
    #
    #
    print("Start")
    print(os.getcwd())

    for building_id in range(101,301):

        #
        # Read in Ausgrid and create results folder
        #

        ausgrid_data = pd.read_csv("data/ausgrid_solar_home_dataset/ausgrid_prosumption.csv", parse_dates=True, index_col=0).resample("h", closed='right').sum()
        os.makedirs(f"results/{building_id}", exist_ok=True)
        #
        # Get the data for building
        #

        building_test_set , dataloader_building_test =  get_data_test(ausgrid_data, str(building_id), MORAI_CONFIG, DATATSET_BOUNDS)

        # 48 hours shift
        pred_df = get_shifted_prediction(building_test_set, 48)

        # save the result in a csv file

        pred_df.to_csv(f"results/{building_id}/naive_shift_48.csv")

        # 168 hours shift

        pred_df = get_shifted_prediction(building_test_set, 168)

        # save the result in a csv file

        pred_df.to_csv(f"results/{building_id}/naive_shift_168.csv")


    print("Done")












