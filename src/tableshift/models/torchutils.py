from typing import Union, Dict, Tuple

import logging
import numpy as np
import rtdl
import scipy
import sklearn
import torch
from tab_transformer_pytorch import TabTransformer
from tqdm import tqdm
from typing import Optional

from ftta.tta import (
    evaluate_tta_with_registry,
    get_predictions_and_labels_tta_with_registry,
)
from tableshift.third_party.saint.models import SAINT


def get_module_attr(model, attr):
    """Get an attribute from (possibly-distributed) module."""
    if hasattr(model, "module"):
        # Case: it is a distributed module; first access model attr explicitly.
        return getattr(model.module, attr)
    else:
        # Case: standard module; fetch the attr.
        return getattr(model, attr)


def unpack_batch(batch: Union[Dict, Tuple[Union[torch.Tensor, None]]]) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, Union[torch.Tensor, None]
]:
    if isinstance(batch, dict):
        # Case: dict-formatted batch; these are used for Ray training.
        x_batch = batch["x"]
        y_batch = batch["y"]
        g_batch = batch["g"]
        d_batch = batch.get("d", None)

    else:
        # Case: tuple of Tensors; these are used for vanilla Pytorch training.
        (x_batch, y_batch, g_batch) = batch[:3]
        d_batch = batch[3] if len(batch) == 4 else None

    return x_batch, y_batch, g_batch, d_batch


def split_num_cat(x, cat_idxs) -> Tuple[torch.Tensor, torch.Tensor]:
    """Use cat_idxs to slice the input batch."""
    if not cat_idxs:
        x_cont = x
        x_categ = None
    else:
        assert max(cat_idxs) < x.shape[1], \
            f"expected cat_idxs in range [0, {x.shape[1] - 1}], got {cat_idxs}"
        categ_mask = np.isin(np.arange(x.shape[1]), cat_idxs)
        x_cont = x[:, ~categ_mask]
        x_categ = x[:, categ_mask]
    return x_cont, x_categ


def apply_model(model: torch.nn.Module, x):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        module = model.module
    else:
        module = model

    if isinstance(module, rtdl.FTTransformer) or isinstance(module, SAINT) \
            or isinstance(module, TabTransformer):
        x_num, x_cat = split_num_cat(x, get_module_attr(module, "cat_idxs"))
        return module(x_num, x_cat)

    else:
        return module(x)

def apply_model_hidden(model: torch.nn.Module, x):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        module = model.module
    else:
        module = model

    if isinstance(module, rtdl.FTTransformer) or isinstance(module, SAINT) \
            or isinstance(module, TabTransformer):
        x_num, x_cat = split_num_cat(x, get_module_attr(module, "cat_idxs"))
        return module.forward_hidden(x_num, x_cat)

    if isinstance(module, rtdl.MLP):
        return module.blocks(x)

    else:
        return module(x)


@torch.no_grad()
def get_predictions_and_labels(model, loader, device=None, as_logits=False) -> Tuple[
    np.ndarray, np.ndarray]:
    """Get the predictions (as logits, or probabilities) and labels."""
    prediction = []
    label = []

    if not device:
        device = f"cuda:{torch.cuda.current_device()}" \
        if torch.cuda.is_available() else "cpu"

    modelname = model.__class__.__name__
    for batch in tqdm(loader, desc=f"{modelname}:getpreds"):
        batch_x, batch_y, _, _ = unpack_batch(batch)
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        outputs = apply_model(model, batch_x)
        prediction.append(outputs)
        label.append(batch_y)
    prediction = torch.cat(prediction).squeeze().cpu().numpy()
    target = torch.cat(label).squeeze().cpu().numpy()
    if not as_logits:
        prediction = scipy.special.expit(prediction)
    return prediction, target

@torch.enable_grad()
def get_predictions_and_labels_tta_ftta(model, loader, device, exp):
    return get_predictions_and_labels_tta_with_registry(
        model=model,
        loader=loader,
        device=device,
        exp=exp,
        method_name="ftta",
    )


def evaluate_tta(
    model,
    loader,
    device,
    split,
    exp: Optional[str] = None,
    tta_method: str = "ftta",
    tent_lr: float = 1e-3,
    tent_steps: int = 1,
    tent_eps: float = 1e-8,
    sar_lr: float = 1e-3,
    sar_steps: int = 1,
    sar_rho: float = 0.05,
    sar_entropy_margin: float = 0.4,
    eata_lr: float = 1e-3,
    eata_steps: int = 1,
    eata_entropy_margin: float = 0.4,
    eata_diversity_margin: float = 0.05,
):
    if split == "train":
        logging.info(f"TTA testing, only ood score will be display, split:{split} skipping")
        return 0
    if split != "ood_test":
        logging.info(f"TTA testing, only ood score will be display, split:{split} skipping")
        return 0

    return evaluate_tta_with_registry(
        model=model,
        loader=loader,
        device=device,
        split=split,
        exp=exp,
        method_name=tta_method,
        method_kwargs={
            "tent_lr": tent_lr,
            "tent_steps": tent_steps,
            "tent_eps": tent_eps,
            "sar_lr": sar_lr,
            "sar_steps": sar_steps,
            "sar_rho": sar_rho,
            "sar_entropy_margin": sar_entropy_margin,
            "eata_lr": eata_lr,
            "eata_steps": eata_steps,
            "eata_entropy_margin": eata_entropy_margin,
            "eata_diversity_margin": eata_diversity_margin,
        },
    )

def evaluate(model, loader, device, split, exp:Optional[str] = None):
    with torch.no_grad():
        model.eval()
        prediction, target = get_predictions_and_labels(model, loader, device)
        prediction = np.round(prediction)
        score = sklearn.metrics.accuracy_score(target, prediction)
    return score
