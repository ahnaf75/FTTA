from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Dict, Optional, Tuple, Type

import copy
import logging
import numpy as np
import scipy
import sklearn
import torch
import torch.nn as nn
from sklearn.metrics import pairwise_distances
from torch.nn import functional as F
from tqdm import tqdm


def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def _get_torchutils_module():
    from tableshift.models import torchutils

    return torchutils


class TtaMethod(nn.Module, ABC):
    @abstractmethod
    def adapt_predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return adapted probabilities/logits for the positive class."""


class FTTAMethod(TtaMethod):
    def __init__(
        self,
        model: nn.Module,
        optimizer_type,
        prior: torch.Tensor,
        lr_list=None,
        device: Optional[str] = None,
        smooth_factor: float = 0.1,
    ):
        super().__init__()
        if lr_list is None:
            lr_list = [1e-4, 5e-4, 1e-5]

        self.base_model1 = deepcopy(model)
        self.base_model2 = deepcopy(model)
        self.base_model3 = deepcopy(model)
        self.model_list = [self.base_model1, self.base_model2, self.base_model3]
        self.optimizer_list = [
            optimizer_type(self.base_model1.parameters(), lr=lr_list[0]),
            optimizer_type(self.base_model2.parameters(), lr=lr_list[1]),
            optimizer_type(self.base_model3.parameters(), lr=lr_list[2]),
        ]
        self.prior = prior
        self.source_y = prior
        self.smooth_factor = smooth_factor
        self.device = device
        if not device:
            self.device = (
                f"cuda:{torch.cuda.current_device()}"
                if torch.cuda.is_available()
                else "cpu"
            )
        logging.info(f"device is {self.device}")

    def suit_neighbors(self, samples, logits, distance):
        sample_distance = pairwise_distances(samples.detach())
        if distance is None:
            distance = sample_distance.mean()
        pseudo_label = logits.argmax(axis=1)
        remain_index = []
        for i, s in enumerate(sample_distance):
            near_neighbor = np.where(s > distance, 0, 1)
            is_remain = ((pseudo_label * near_neighbor).sum()) / near_neighbor.sum()
            if abs(is_remain - pseudo_label[i]) > 0.3:
                continue
            remain_index.append(i)
        return torch.tensor(remain_index)

    @torch.enable_grad()
    def online_logits(self, out_list):
        factor = torch.zeros(size=(1, 3), dtype=torch.float32)
        for i, out in enumerate(out_list):
            factor[0][i] = 1 - (softmax_entropy(out) * abs(out[:, 1] - out[:, 0])).mean()
        factor = F.normalize(factor, p=1, dim=1)
        return (
            out_list[0] * factor[0][0]
            + out_list[1] * factor[0][1]
            + out_list[2] * factor[0][2]
        )

    def forward(self, x):
        torchutils = _get_torchutils_module()
        out_logits = []
        for model in self.model_list:
            outputs = torchutils.apply_model(model, x)
            two_shape = 1 - torch.sigmoid(outputs)
            outputs = torch.cat([two_shape, torch.sigmoid(outputs)], dim=1)
            out_logits.append(outputs)

        final = self.online_logits(out_logits)
        logits_p = F.normalize(final * self.prior / self.source_y, p=1)

        remain_index = self.suit_neighbors(x.cpu(), final.cpu(), None).to(self.device)
        for logits_back, optimizer in zip(out_logits, self.optimizer_list):
            if len(remain_index) > 0:
                loss = (
                    softmax_entropy(torch.index_select(logits_back, dim=0, index=remain_index))
                    * torch.index_select(
                        abs(logits_back[:, 0] - logits_back[:, 1]),
                        dim=0,
                        index=remain_index,
                    )
                ).mean(0)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        y_hat = logits_p.argmax(axis=1)
        condition = logits_p[
            softmax_entropy(logits_p)
            < softmax_entropy(torch.tensor([[0.7, 1 - 0.7]]).to(self.device))
        ]
        A_pre = torch.tensor(
            [
                torch.mean(logits_p[y_hat == 0], dim=0).detach().cpu().numpy(),
                torch.mean(logits_p[y_hat == 1], dim=0).detach().cpu().numpy(),
            ]
        ).to(self.device)
        B_acc = torch.mean(condition, dim=0)
        prior_fac = torch.linalg.inv(A_pre) @ B_acc
        if not torch.any(torch.isnan(prior_fac)):
            self.prior = F.softmax((self.prior - self.smooth_factor * prior_fac), dim=0)

        adapt_samples = torch.where(abs(final[:, 0] - final[:, 1]) > 0, 1, 0)
        final[adapt_samples == 1] = F.normalize(
            final[adapt_samples == 1] * self.prior / self.source_y, p=1
        )

        return final[:, 1].detach()

    def adapt_predict(self, x: torch.Tensor) -> torch.Tensor:
        return self(x)


# Backward-compatible class name.
class FTTA(FTTAMethod):
    pass


DATASET_PRIOR_PROBS: Dict[str, Tuple[float, float]] = {
    "anes": (1 - 0.691, 0.691),
    "heloc": (1 - 0.24, 0.24),
    "assistments": (1 - 0.694, 0.694),
    "diabetes_readmission": (1 - 0.42, 0.42),
    "brfss_blood_pressure": (1 - 0.403, 0.403),
}


def get_source_prior(exp: str, device: str) -> torch.Tensor:
    if exp not in DATASET_PRIOR_PROBS:
        print("please check exp name")
        raise ValueError("exp wrong")
    return torch.tensor(DATASET_PRIOR_PROBS[exp]).to(device)


class TtaRegistry:
    def __init__(self):
        self._registry: Dict[str, Type[TtaMethod]] = {}

    def register(self, name: str, method_cls: Type[TtaMethod]) -> None:
        self._registry[name] = method_cls

    def create(self, name: str, **kwargs) -> TtaMethod:
        if name not in self._registry:
            raise ValueError(f"unknown tta method {name}")
        return self._registry[name](**kwargs)


TTA_REGISTRY = TtaRegistry()
TTA_REGISTRY.register("ftta", FTTAMethod)


@torch.no_grad()
def get_predictions_and_labels_unadapt(
    model,
    loader,
    device=None,
    as_logits: bool = False,
):
    prediction = []
    label = []

    if not device:
        device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"

    torchutils = _get_torchutils_module()
    modelname = model.__class__.__name__
    for batch in tqdm(loader, desc=f"{modelname}:getpreds"):
        batch_x, batch_y, _, _ = torchutils.unpack_batch(batch)
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        outputs = torchutils.apply_model(model, batch_x)
        prediction.append(outputs)
        label.append(batch_y)

    prediction = torch.cat(prediction).squeeze().cpu().numpy()
    target = torch.cat(label).squeeze().cpu().numpy()
    if not as_logits:
        prediction = scipy.special.expit(prediction)
    return prediction, target


@torch.enable_grad()
def get_predictions_and_labels_tta_with_registry(
    model,
    loader,
    device,
    exp,
    method_name: str = "ftta",
):
    optimizer = torch.optim.SGD
    prediction = []
    label = []

    source_y = get_source_prior(exp, device)

    tta_method = TTA_REGISTRY.create(
        method_name,
        model=model,
        optimizer_type=optimizer,
        prior=source_y,
        lr_list=[1e-5, 5e-4, 1e-4],
        device=device,
    )

    torchutils = _get_torchutils_module()
    modelname = model.__class__.__name__
    for batch in tqdm(loader, desc=f"{modelname}:getpreds"):
        batch_x, batch_y, _, _ = torchutils.unpack_batch(batch)
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        outputs = tta_method.adapt_predict(batch_x)
        prediction.append(outputs)
        label.append(batch_y)

    prediction = torch.cat(prediction).squeeze().cpu().numpy()
    target = torch.cat(label).squeeze().cpu().numpy()
    return prediction, target


def evaluate_tta_with_registry(
    model,
    loader,
    device,
    split,
    exp: Optional[str] = None,
    method_name: str = "ftta",
):
    if split == "train":
        logging.info(f"TTA testing, only ood score will be display, split:{split} skipping")
        return 0

    if split == "ood_test":
        with torch.enable_grad():
            model_ense = copy.deepcopy(model)
            model_ense.train()
            pre, tar = get_predictions_and_labels_tta_with_registry(
                model_ense,
                loader,
                device,
                exp,
                method_name=method_name,
            )
            pre = np.round(pre)
            score = sklearn.metrics.accuracy_score(tar, pre)
            print("\n", "FTTA: acc ", "\n", score, "\n")

        with torch.no_grad():
            model.eval()
            pre, tar = get_predictions_and_labels_unadapt(model, loader, device)
            pre = np.round(pre)
            score = sklearn.metrics.accuracy_score(tar, pre)
            print("\n", "Unadapt: acc ", "\n", score, "\n")
    else:
        logging.info(f"TTA testing, only ood score will be display, split:{split} skipping")
        return 0

    return score
