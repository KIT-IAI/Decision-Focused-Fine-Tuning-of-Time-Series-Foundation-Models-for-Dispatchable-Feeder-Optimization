from src.datasets.datasets import DFRTrainableBuildingDataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import torch




def get_data_test(ausgrid_data, building_id, dataset_bounds):
    """
    Get the test data for a specific building.
        Args:
            :param ausgrid_data: containing the Ausgrid data.
            :param building_id: The ID of the building to get the data for.
            :param dataset_bounds: The bounds of the dataset.

        Returns:
            return: The building dataset and the dataloader for the test data.
    """

    # The dataset of building
    building = DFRTrainableBuildingDataset(ausgrid_data[str(building_id)],*dataset_bounds["test"],building_id=int(building_id), dataset="test", verbose=True)
    # dataloader
    dataloader = DataLoader(building, batch_size=32, shuffle=False)
    return building, dataloader

# save csv file method which catches the existance and increments a counter if already exists
def save_csv_file(df, path):
    """
    Save a pandas DataFrame to a CSV file, incrementing the filename if it already exists.
        Args:
            :param df: The DataFrame to save.
            :param path: The base path for the CSV file.

        Returns:
            :return: The path to the saved CSV file.
    """
    counter = 0
    path = path + f"_run_{counter}.csv"
    while True:
        try:
            df.to_csv(path, mode='x')
            break
        except FileExistsError:
            old_counter = counter
            counter += 1
            path = path.replace(f"_run_{old_counter}.csv", f"_run_{counter}.csv")
    return path


def evaluate_model(model, building_dataset, dataloader, building_id, string_evaluation, morai_config):
    """"
    Evaluate the model on the test data and save the results to a CSV file.
        Args:
            :param model: The model to evaluate.
            :param building_dataset: The dataset of the building.
            :param dataloader: The dataloader of the building.
            :param building_id: The ID of the building.
            :param string_evaluation: A string to identify the evaluation. E.g. local_surrogate_epochs_5_dora_simulated 
            :param morai_config: The configuration for the Morai model.
        Returns:
            :return: The predictions as a pandas DataFrame.
            
    """

    pdt = morai_config["pdt"]
    patch_size = morai_config["patch_size"]

    # get the prediction

    pred = []
    model.eval()
    
    for batch in dataloader:
        

        batch[0]["past_target"] = batch[0]["past_target"].to(dtype=torch.bfloat16).to(device="cuda")
        batch[0]["past_observed_target"] = batch[0]["past_observed_target"].to(device="cuda")
        batch[0]["past_is_pad"] = batch[0]["past_is_pad"].to(device="cuda")

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred_tmp = model._get_distr(patch_size,**batch[0])  
           

        pred_tmp = model._format_preds(patch_size, pred_tmp.mean,1)
        pred_tmp = pred_tmp.detach().cpu().numpy()
        pred.append(pred_tmp)

    pred = np.concatenate(pred, axis=0) 

    # get all indexes of pred
    indexes = [building_dataset.get_time_by_index(i) for i in range(len(building_dataset))]

    # create a pandas dataframe
    pred_df = pd.DataFrame(pred, index=indexes, columns=[f"{i}" for i in range(0, pdt)])

    # save the result in a csv file

    # rename colum o to time
    pred_df.index.name = "time"
    
    # save the csv file
    path = save_csv_file(pred_df, f"results/{building_id}/{string_evaluation}")

    return pred_df

def save_gt_from_pred_df(building_test_set, pred_df, morai_config, id):
    """
    Save the ground truth from the prediction DataFrame to a CSV file.
        Args:
            :param building_test_set: The test dataset of the building.
            :param pred_df: The DataFrame containing the predictions.
            :param morai_config: The configuration for the Morai model.
            :param id: The ID of the building to save the ground truth for.
    """
    pdt = morai_config["pdt"]
    # get all indexes of pred
    indexes = [building_test_set.get_time_by_index(i) for i in range(len(building_test_set))]
    
    gt = []
    for i in range(len(building_test_set)):
        gt.append(building_test_set.get_gt(i).reshape(1,-1))
    gt = np.concatenate(gt, axis=0)
    gt_df = pd.DataFrame(gt, index=indexes, columns=[f"{i}" for i in range(0, pdt)])

    # save index in gt_df which is contained in pred_df
    gt_df = gt_df[gt_df.index.isin(pred_df.index)]
    gt_df.index.name = "time"
    gt_df.to_csv(f"results/{id}/gt.csv")
