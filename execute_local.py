import pandas as pd
import os
import argparse
import copy

from src.models.morai import morai_refresh
from src.trainings.training_morai import training
from src.utils import get_data_test, evaluate_model , save_gt_from_pred_df
from src.configuration import TRAINING_CONFIG, MORAI_CONFIG , LORA_CONFIG, DORA_CONFIG


def _execute_config(tmp_training_config, args_low, args_high, peft, peft_epochs, soe, loss):
    """
    Execute the configuration

    Args:
        tmp_training_config: dict, the training configuration
        args_low: int, the lowest building id to evaluated in the evaluation loop
        args_high: int, the highest building id to evaluated in the evaluation loop
        peft: str, peft method
        peft_epochs: int, the number of peft epochs
        soe: str, soe to use

    Returns:
        None

    """


    #
    # Refresh the model
    #
    model = morai_refresh(MORAI_CONFIG)

    training_low = tmp_training_config["training_building_low"]
    training_high = tmp_training_config["training_building_high"]

    #
    # Interchange the configuration for everything what should be explorated
    #
    
    # set loss
    tmp_training_config["loss"] = loss
    # set soe
    tmp_training_config["soe"] = soe

    # exchange the peft config
    if peft == "dora":
        print("Dora")
        tmp_training_config["peft_config"] = DORA_CONFIG
    elif peft == "lora":
        print("Lora")
        tmp_training_config["peft_config"] = LORA_CONFIG
    else:
        print("not configurated")
        Exception("Not configurated")

    # set peft epochs
    tmp_training_config["peft_epochs"] = peft_epochs
        
    # train the model

    model = training(model, ausgrid_data, range(training_low, training_high), tmp_training_config)
    
    #
    # generate the string unique for the run
    # 

    run_string = f"local_{loss}_epochs_{peft_epochs}_{peft}_{soe}"

    for building_id in range(args_low, args_high):
        building_test_set , dataloader_building_test =  get_data_test(ausgrid_data, str(building_id), tmp_training_config["dataset_bounds"])
        pred_df = evaluate_model(model, building_test_set, dataloader_building_test, str(building_id), run_string , MORAI_CONFIG)




#main method
if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-low", type=int, help="Lowest building id to evaluate", default=101)
    parser.add_argument("-high", type=int, help="Highest building id to evaluate", default=301)
    parser.add_argument("-peft", type=str, help="Peft Method", default="peft")
    parser.add_argument("-peft_epochs", type=int, help="Peft epochs", default=1)
    parser.add_argument("-loss", type=str, help="Loss", default="mse")
    parser.add_argument("-soe", type=str, help="Using SoE", default="simulated")

    args = parser.parse_args()

    # Print the args
    print("Code executed with: ", args)
    # print file name
    print("File name: ", __file__)
    

    # create results folder for all buildings containing in the range low high

    for i in range(args.low, args.high):
        if not os.path.exists(f"results/{i}"):
            os.makedirs(f"results/{i}")

    #
    # Read in Ausgrid Data
    #

    ausgrid_data = pd.read_csv("data/ausgrid_solar_home_dataset/ausgrid_prosumption.csv", parse_dates=True, index_col=0).resample("h", closed='right').sum()

    #
    # Loop over all buildings and execute the configuration
    #
 
    for i in range(args.low, args.high):

        #
        # This is the only training configuration change which is done outside of execute config to exchange the training buildings into a local setting
        #

        tmp_training_config = copy.deepcopy(TRAINING_CONFIG)

        # replace training buildings with i th building
        tmp_training_config["training_building_low"] = i
        tmp_training_config["training_building_high"] = i+1

        _execute_config(tmp_training_config, i, i+1, args.peft, args.peft_epochs, args.soe,args.loss)

    print("Done")












