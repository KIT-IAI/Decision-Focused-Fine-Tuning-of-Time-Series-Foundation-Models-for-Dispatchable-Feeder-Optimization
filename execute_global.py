import pandas as pd
import os
import argparse
import copy

from src.models.morai import morai_refresh
from src.trainings.training_morai import training
from src.utils import get_data_test, evaluate_model , save_gt_from_pred_df
from src.configuration import TRAINING_CONFIG, MORAI_CONFIG , LORA_CONFIG, DORA_CONFIG



def _execute_config(tmp_training_configuration, args_low, args_high, peft, peft_epochs, gt_gen, no_training, soe, loss):
    """
    Execute the configuration

    Args:
        tmp_training_config: dict, the training configuration
        args_low: int, the lowest building id to evaluated in the evaluation loop
        args_high: int, the highest building id to evaluated in the evaluation loop
        peft: str, peft method
        peft_epochs: int, the number of peft epochs
        gt_gen: bool, if true generate the ground truth
        no_training: bool, if true do not train the model
        soe: str , soe to use
        loss: str, loss function

    Returns:
        None
    """

    #
    # Refresh the model
    #
    model = morai_refresh(MORAI_CONFIG)

    training_low = tmp_training_configuration["training_building_low"]
    training_high = tmp_training_configuration["training_building_high"]

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



    #
    # generate the string unique for the run
    # 

    run_string = f"global_{loss}_epochs_{peft_epochs}_{peft}_{soe}"

    #
    # train the model
    # 

    if not no_training:
        model = training(model, ausgrid_data, range(training_low, training_high), tmp_training_config)
    else:    
        run_string = f"global_no_training"
        
    #
    # evaluate the model
    #     

    for building_id in range(args_low, args_high):
        building_test_set , dataloader_building_test =  get_data_test(ausgrid_data, str(building_id), tmp_training_config["dataset_bounds"])
        pred_df = evaluate_model(model, building_test_set, dataloader_building_test, str(building_id), run_string, MORAI_CONFIG )

        #
        # generate the ground truth for all buildings
        #

        if gt_gen:
            save_gt_from_pred_df(building_test_set, pred_df, MORAI_CONFIG, building_id)




#main method
if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-low", type=int, help="Lowest building id to evaluate", default=101)
    parser.add_argument("-high", type=int, help="Highest building id to evaluate", default=301)
    parser.add_argument("-peft", type=str, help="Peft Method", default="dora")
    parser.add_argument("-peft_epochs", type=int, help="Peft epochs", default=1)
    parser.add_argument("-loss", type=str, help="Loss", default="mse")
    parser.add_argument("-no_training", type=bool, help="no_training", default=False)
    parser.add_argument("-gt_gen", type=bool, help="gt_gen", default=False)
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
    # Execute the configuration
    #
 
    tmp_training_config = copy.deepcopy(TRAINING_CONFIG)
    _execute_config(tmp_training_config, args.low, args.high, args.peft, args.peft_epochs, args.gt_gen, args.no_training, args.soe, args.loss)
    print("Done")










