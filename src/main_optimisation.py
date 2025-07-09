from optimisation.ClassOpti import Opti
import argparse
import pandas as pd
from glob import glob
import os
import pickle
import numpy as np
import wandb


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def pickle_opimisation(optimisation, path):
    """
    Args:
        optimisation: optimisation instance
        path: path to safe the file contains res_path and the given forecast_path

    Returns: nothing

    """
    # open a file, where you ant to store the data
    file = open(path, 'wb')

    # dump information to that file
    pickle.dump(optimisation, file)

    # close the file
    file.close()

#
#
#

START_DATE = pd.to_datetime('2012-07-08')
END_DATE = pd.to_datetime('2013-06-28')


#
#
#

if __name__ == '__main__':
    #
    # Argument Parser
    #

    


    parser = argparse.ArgumentParser(description="Parser")
    parser.add_argument('-id', type=str, help='building_id', default="101")
    
    # argument for the name regarding the wohle run
    parser.add_argument('-regex', type=str,  default="*")
    args = parser.parse_args()


    config = {
        "id": args.id,
        "regex": args.regex,
    }

    wandb.init(
    # set the wandb project where this run will be logged
    project="OWN PROJECT NAME",
    
    # track hyperparameters and run metadata

    config=config,
    )

    #
    # Create result folder if not existent
    #

    optimisation_path = f"results_optimisation/{args.id}"
    make_dir(optimisation_path)

    #
    # read in the specific file depending on the id and regex given
    #

    forecast_paths = glob(f"results/{args.id}/{args.regex}.csv")
    
    #
    # TODO Change the path to a path which not needed to be created in advance
    # 
    actual_path_test = glob(f"data/ausgrid_solar_home_dataset/daily/gt_daily_{args.id}.csv")


    ds_costs_optimisation = { 'test': pd.DataFrame()}
    imbalance_costs_optimisation = { 'test': pd.DataFrame()}

    print(forecast_paths)
    strings_runs= []
    # Optimisation
    for forecast_path in forecast_paths:
        str_fc = forecast_path.split("/")[-1].split(".")[0]
        strings_runs.append(str_fc)

        optimisation = Opti(1)


        optimisation.run_deterministic_different_forecasts([optimisation.e_max["house0"] / 2], [forecast_path], [forecast_path], actual_path_test, START_DATE,
                                     END_DATE)


        # Set string for saving
        str_dataset = 'test'


        pickle_opimisation(optimisation, optimisation_path +"/"+ str_fc + ".pkl")


        optimisation.calculate_ds_costs_daily('house0')
        optimisation.calculate_imbalance_costs_daily('house0')

        # Saving costs in dict


        ds_costs_optimisation[str_dataset] = pd.concat(
            [ds_costs_optimisation[str_dataset], optimisation.ds_costs_daily.rename(str_fc)],
            axis=1)
        imbalance_costs_optimisation[str_dataset] = pd.concat(
            [imbalance_costs_optimisation[str_dataset], optimisation.imbalance_costs_daily.rename(str_fc)], axis=1)
        
    wandb.log({"ds_costs_optimisation": ds_costs_optimisation["test"], "imbalance_costs_optimisation": imbalance_costs_optimisation["test"]})
    for string in strings_runs:
        wandb.log({f"ds_costs_optimisation_{string}": ds_costs_optimisation["test"][string].values.mean()})
        wandb.log({f"imbalance_costs_optimisation_{string}": imbalance_costs_optimisation["test"][string].values.mean()})
        total_costs = ds_costs_optimisation["test"][string].values + 10 * imbalance_costs_optimisation["test"][string].values
        wandb.log({f"total_costs_{string}": total_costs.mean()})
    # Saving cost dicts
    path_ds = f"{optimisation_path}/ds_costs_daily_optimisation"
    path_imbalance = f"{optimisation_path}/imbalance_costs_daily_optimisation"

    make_dir(path_ds)
    make_dir(path_imbalance)

    with open(path_ds+ f"/ds_{args.regex}.pkl", 'wb') as file:
        pickle.dump(ds_costs_optimisation, file)
    with open(path_imbalance + f"/imb_{args.regex}.pkl", 'wb') as file:
        pickle.dump(imbalance_costs_optimisation, file)

    wandb.finish()








