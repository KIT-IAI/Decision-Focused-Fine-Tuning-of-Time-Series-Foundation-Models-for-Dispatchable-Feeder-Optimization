from uni2ts.model.moirai import MoiraiForecast
from sktime.libs.uni2ts.forecast import MoiraiForecast as SktimeMoiraiForecast
from huggingface_hub import hf_hub_download
from torch import nn


def morai_refresh(morai_config):
    """
    Refresh the Moirai model with the given configuration.
    Args:
        morai_config (dict): Configuration dictionary containing model parameters.
    Returns:
        MoiraiForecast: The refreshed Moirai model.
    """

    # read in from config
    pdt = morai_config["pdt"]
    ctx = morai_config["ctx"]
    psz = morai_config["patch_size"]
    size = morai_config["size"]
    
    # 
    # Due to licensing issues, we cannot use the original Moirai model from uni2ts.
    # Instead, we use the sktime version of Moirai, which is a modified version
    # of the original Moirai model that is compatible with sktime.
    # The sktime version of Moirai is available on Hugging Face Hub.
    #  

    checkpoint = hf_hub_download(
                    repo_id=f"sktime/moirai-1.0-R-{size}", filename="model.ckpt"
                )

    sktime_moirai_module = SktimeMoiraiForecast.load_from_checkpoint(
    **{"checkpoint_path": checkpoint},
    **{'prediction_length': pdt, 'context_length': ctx, 'patch_size': psz, 
                                         'num_samples': 100, 'target_dim': 1, 'feat_dynamic_real_dim': 0, 'past_feat_dynamic_real_dim': 0, 
                                      }
    ).module

    weight, bias = sktime_moirai_module.param_proj.proj.weights_logits.parameters()
    weight = weight[[3, 4, 0, 1, 2]]
    bias = bias[[3, 4, 0, 1, 2]]
    sktime_moirai_module.param_proj.proj.weights_logits.weight = nn.Parameter(weight)
    sktime_moirai_module.param_proj.proj.weights_logits.bias = nn.Parameter(bias)

    model = MoiraiForecast(**{'prediction_length': pdt, 'context_length': ctx, 'patch_size': psz, 
                                            'num_samples': 100, 'target_dim': 1, 'feat_dynamic_real_dim': 0, 'past_feat_dynamic_real_dim': 0, 
                                        },
                                        module=sktime_moirai_module,)


    model.to("cuda")

    return model