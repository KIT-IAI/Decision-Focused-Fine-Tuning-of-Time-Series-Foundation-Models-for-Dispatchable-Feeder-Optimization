
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW, RAdam
import torch
from peft import LoraConfig
from peft import get_peft_model

# Imports 
from src.models.optimisation_model import OptiEstimator
from src.datasets.datasets import DFRTrainableBuildingDataset




def _loss(pred, target, cost_est, loss):
    if loss == "mse":
        loss = torch.nn.MSELoss()(pred, target)

    elif loss == "mae":
        loss = torch.nn.L1Loss()(pred, target)

    elif loss == "surrogate":
        loss = cost_est
        
    else:
        raise ValueError("Loss function not implemented: " + loss)
    return loss


def training(model, ausgrid_data, train_building_range, training_config):
    """
    Train the model with the given training configuration.
    
    Args:
        :param model: The model to train.
        :param ausgrid_data: The Ausgrid data.
        :param train_building_range: The range of buildings to train on.
        :param training_config: The training configuration.
    Returns:
        :return: The trained model.
    """


    loss_config = training_config["loss"]
    ctx = training_config["ctx"]
    pdt = training_config["pdt"]
    patch_size = training_config["patch_size"]
    batch_size = training_config["batch_size"]
    peft_epochs = training_config["peft_epochs"]
    dataset_bounds = training_config["dataset_bounds"]
    peft_config = training_config["peft_config"]
    optimizer = training_config["optimizer"]
    op_est_names = training_config["op_est_names"]
    soe = training_config["soe"]

    print(f"Training with Training Config: {training_config}")


    # initalize lora model
    peft_config = LoraConfig(**peft_config)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()



    # load op models
    op_models = []
    for op_model in op_est_names:
        tmp_model = OptiEstimator(1)
        tmp_model.load_state_dict(torch.load("models/surrogate/" + op_model))
        tmp_model.eval()
        tmp_model.to("cuda")
        # freeze model
        for param in tmp_model.parameters():
            param.requires_grad = False
        print(f"Loaded surrogate model {op_model} and freezed it")

        op_models.append(tmp_model)
    
    # create a trainable dataset for building 
    
    building = torch.utils.data.ConcatDataset([DFRTrainableBuildingDataset(ausgrid_data[str(building_id)],*dataset_bounds["train"],building_id=int(building_id), dataset="train", soe=soe) for building_id in train_building_range])
    building_val = torch.utils.data.ConcatDataset([DFRTrainableBuildingDataset(ausgrid_data[str(building_id)],*dataset_bounds["validation"],building_id=int(building_id), dataset="validation", soe=soe) for building_id in train_building_range])

    if optimizer == "AdamW":
        optimizer = AdamW(model.parameters(), lr=0.0001)
    else:
        raise ValueError("Optimizer not implemented")
        
    scaler  = torch.cuda.amp.GradScaler()


    dataloader_train = DataLoader(building, batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(building_val, batch_size=len(building_val))


    for epoch in range(peft_epochs):

        
        
        for i, batch in enumerate(dataloader_train):

            batch[0]["past_target"] = batch[0]["past_target"].to(device="cuda", dtype=torch.bfloat16)
            batch[0]["past_observed_target"] = batch[0]["past_observed_target"].to(device="cuda")
            batch[0]["past_is_pad"] = batch[0]["past_is_pad"].to(device="cuda")
            

            target = batch[1].to(device="cuda", dtype=torch.bfloat16)
            soe_stats = batch[2].to(device="cuda", dtype=torch.bfloat16)

            model.train()
            # Zero gradients
            optimizer.zero_grad()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Forward pass
                
                pred = model._get_distr(
                            patch_size=patch_size,**batch[0]
                )
                pred = model._format_preds(patch_size, pred.mean, 1)

                # Surrogate cost estimation
                cost_est = []
                for op_model in op_models:
                    cost_est.append(op_model(torch.cat([pred, target,soe_stats], dim=1)))
                cost_est = torch.stack(cost_est, dim=1).mean()
            

                # Calculate loss and backpropagate

                loss = _loss(pred, target, cost_est, loss_config)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


            # Validation routine every 50 steps including the first step
            if i % 50 == 0:
                model.eval()
                
                val_loss = None
                val_loss_list = []

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    with torch.no_grad():
                        for batch in dataloader_val:
                            batch[0]["past_target"] = batch[0]["past_target"].to(device="cuda", dtype=torch.bfloat16)
                            batch[0]["past_observed_target"] = batch[0]["past_observed_target"].to(device="cuda")
                            batch[0]["past_is_pad"] = batch[0]["past_is_pad"].to(device="cuda")

                            target = batch[1].to(device="cuda", dtype=torch.bfloat16)
                            soe_stats = batch[2].to(device="cuda", dtype=torch.bfloat16)

                            # Forward pass
                            pred = model._get_distr(patch_size,**batch[0])    
                            pred = model._format_preds(patch_size, pred.mean,1)

                            # Surrogate cost estimation
                            cost_est = []
                            for op_model in op_models:
                                cost_est.append(op_model(torch.cat([pred, target, soe_stats], dim=1)))
                            cost_est = torch.stack(cost_est, dim=1).mean()
                            
                            # Calculate loss
                            val_loss = _loss(pred, target, cost_est, loss_config).detach().cpu()
                            val_loss_list.append(val_loss)
                            
                
                
                val_loss = torch.stack(val_loss_list).mean()
                print(f"Epoch {epoch} - Validation Loss: {val_loss.mean()}")
    
    model.eval()
    return model
