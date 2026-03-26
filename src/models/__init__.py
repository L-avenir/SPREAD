from typing import *
from torch import Tensor
from torch.optim import Optimizer

from functools import partial

from torch import optim

from .sg2sc_cast_diffusion_fixedmiche import Sg2ScCASTDiffusion_fixed_michelangelo

def model_from_config(
    config: Dict[str, Any],
    num_objs=-1, num_preds=-1, num_regions=-1,
    text_emb_dim=512,
    **kwargs
):
    if "sg2sc" in config["name"]:
        if "cast" in config["name"]:
            if 'fixed' in config["name"]:
                print("Using simple Michelangelo CAST Diffusion with point")
                return Sg2ScCASTDiffusion_fixed_michelangelo(
                    num_regions, num_preds, config,
                    use_objfeat="objfeat" in config["name"]
                )
        else:
            raise NotImplementedError(f"Unknown model name: {config['name']}")

    else:
        raise NotImplementedError(f"Unknown model name: {config['name']}")


def optimizer_from_config(config: Dict[str, Any], params: Iterable[Tensor]) -> Optimizer:
    name = config["name"]
    lr = config["lr"]
    weight_decay = config.get("weight_decay", 0.)

    kwargs = dict(
        lr=lr,
        weight_decay=weight_decay,
        capturable = config.get("capturable", False),
        fused = config.get("fused", False),
        foreach = config.get("foreach", False),
    )

    momentum = config.get("momentum", 0.)
    nesterov = config.get("nesterov", False)

    betas = config.get("betas", (0.9, 0.999))
    
    print(f"Using optimizer: {name} with config: {kwargs}")

    optimizer = {
        "sgd": partial(optim.SGD, momentum=momentum, nesterov=nesterov, **kwargs),
        "adam": partial(optim.Adam, betas=betas, **kwargs),
        "adamw": partial(optim.AdamW, betas=betas, **kwargs),
        "radam": partial(optim.RAdam, betas=betas, **kwargs),
    }[name]

    return optimizer(params)
