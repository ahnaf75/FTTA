from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Type

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


def _binary_probs_from_outputs(outputs: torch.Tensor) -> torch.Tensor:
    pos_probs = torch.sigmoid(outputs)
    if pos_probs.ndim == 1:
        pos_probs = pos_probs.unsqueeze(1)
    return torch.cat([1 - pos_probs, pos_probs], dim=1)


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
            outputs = _binary_probs_from_outputs(outputs)
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


def _collect_tent_params(model: nn.Module) -> List[nn.Parameter]:
    norm_params: List[nn.Parameter] = []
    for module in model.modules():
        if isinstance(
            module,
            (
                nn.BatchNorm1d,
                nn.BatchNorm2d,
                nn.BatchNorm3d,
                nn.LayerNorm,
                nn.GroupNorm,
                nn.InstanceNorm1d,
                nn.InstanceNorm2d,
                nn.InstanceNorm3d,
            ),
        ):
            if getattr(module, "weight", None) is not None and module.weight.requires_grad:
                norm_params.append(module.weight)
            if getattr(module, "bias", None) is not None and module.bias.requires_grad:
                norm_params.append(module.bias)

    if norm_params:
        return norm_params
    return [p for p in model.parameters() if p.requires_grad]


class TENTMethod(TtaMethod):
    def __init__(
        self,
        model: nn.Module,
        optimizer_type,
        tent_lr: float = 1e-3,
        tent_steps: int = 1,
        tent_eps: float = 1e-8,
        device: Optional[str] = None,
        **_: Any,
    ):
        super().__init__()
        self.model = deepcopy(model)
        self.model.train()
        self.device = device or (
            f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.tent_steps = max(1, int(tent_steps))
        self.tent_eps = float(tent_eps)
        params = _collect_tent_params(self.model)
        self.optimizer = optimizer_type(params, lr=float(tent_lr))
        logging.info(f"TENT device is {self.device}")

    @torch.enable_grad()
    def adapt_predict(self, x: torch.Tensor) -> torch.Tensor:
        torchutils = _get_torchutils_module()
        x = x.to(self.device)
        probs = None
        for _ in range(self.tent_steps):
            outputs = torchutils.apply_model(self.model, x)
            probs = _binary_probs_from_outputs(outputs)
            entropy = -(probs * torch.log(probs.clamp_min(self.tent_eps))).sum(dim=1).mean()
            self.optimizer.zero_grad()
            entropy.backward()
            self.optimizer.step()

        assert probs is not None
        return probs[:, 1].detach()


class SARMethod(TtaMethod):
    def __init__(
        self,
        model: nn.Module,
        optimizer_type,
        sar_lr: float = 1e-3,
        sar_steps: int = 1,
        sar_rho: float = 0.05,
        sar_entropy_margin: float = 0.4,
        tent_eps: float = 1e-8,
        device: Optional[str] = None,
        **_: Any,
    ):
        super().__init__()
        self.model = deepcopy(model)
        self.model.train()
        self.device = device or (
            f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.sar_steps = max(1, int(sar_steps))
        self.sar_rho = float(sar_rho)
        self.sar_entropy_margin = float(sar_entropy_margin)
        self.eps = float(tent_eps)
        params = _collect_tent_params(self.model)
        self.optimizer = optimizer_type(params, lr=float(sar_lr))
        logging.info(f"SAR device is {self.device}")

    @torch.enable_grad()
    def adapt_predict(self, x: torch.Tensor) -> torch.Tensor:
        torchutils = _get_torchutils_module()
        x = x.to(self.device)
        probs = None
        for _ in range(self.sar_steps):
            outputs = torchutils.apply_model(self.model, x)
            probs = _binary_probs_from_outputs(outputs)
            ent = -(probs * torch.log(probs.clamp_min(self.eps))).sum(dim=1)
            reliable_mask = ent < self.sar_entropy_margin
            if reliable_mask.any():
                loss = ent[reliable_mask].mean()
            else:
                loss = ent.mean()
            self.optimizer.zero_grad()
            loss.backward()
            grad_list = [torch.norm(p.grad.detach(), p=2) for p in self.model.parameters() if p.grad is not None]
            if not grad_list:
                continue
            grad_norm = torch.norm(torch.stack(grad_list), p=2)
            scale = self.sar_rho / (grad_norm + 1e-12)
            perturbations: List[Tuple[nn.Parameter, torch.Tensor]] = []
            for p in self.model.parameters():
                if p.grad is not None:
                    e_w = p.grad * scale
                    p.data.add_(e_w)
                    perturbations.append((p, e_w))

            outputs_sam = torchutils.apply_model(self.model, x)
            probs_sam = _binary_probs_from_outputs(outputs_sam)
            ent_sam = -(probs_sam * torch.log(probs_sam.clamp_min(self.eps))).sum(dim=1)
            if reliable_mask.any():
                second_loss = ent_sam[reliable_mask].mean()
            else:
                second_loss = ent_sam.mean()
            self.optimizer.zero_grad()
            second_loss.backward()

            for p, e_w in perturbations:
                p.data.sub_(e_w)
            self.optimizer.step()

        assert probs is not None
        return probs[:, 1].detach()


class EATAMethod(TtaMethod):
    def __init__(
        self,
        model: nn.Module,
        optimizer_type,
        eata_lr: float = 1e-3,
        eata_steps: int = 1,
        eata_entropy_margin: float = 0.4,
        eata_diversity_margin: float = 0.05,
        tent_eps: float = 1e-8,
        device: Optional[str] = None,
        **_: Any,
    ):
        super().__init__()
        self.model = deepcopy(model)
        self.model.train()
        self.device = device or (
            f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.eata_steps = max(1, int(eata_steps))
        self.entropy_margin = float(eata_entropy_margin)
        self.diversity_margin = float(eata_diversity_margin)
        self.eps = float(tent_eps)
        self.running_prob: Optional[torch.Tensor] = None
        params = _collect_tent_params(self.model)
        self.optimizer = optimizer_type(params, lr=float(eata_lr))
        logging.info(f"EATA device is {self.device}")

    @torch.enable_grad()
    def adapt_predict(self, x: torch.Tensor) -> torch.Tensor:
        torchutils = _get_torchutils_module()
        x = x.to(self.device)
        probs = None
        for _ in range(self.eata_steps):
            outputs = torchutils.apply_model(self.model, x)
            probs = _binary_probs_from_outputs(outputs)
            ent = -(probs * torch.log(probs.clamp_min(self.eps))).sum(dim=1)
            reliable_mask = ent < self.entropy_margin

            if self.running_prob is None:
                diverse_mask = torch.ones_like(reliable_mask, dtype=torch.bool)
            else:
                cosine_sim = F.cosine_similarity(probs.detach(), self.running_prob.unsqueeze(0), dim=1)
                diverse_mask = (1 - cosine_sim) > self.diversity_margin

            selected_mask = reliable_mask & diverse_mask
            if selected_mask.any():
                selected_probs = probs[selected_mask]
                loss = -(selected_probs * torch.log(selected_probs.clamp_min(self.eps))).sum(dim=1).mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                with torch.no_grad():
                    batch_mean = selected_probs.mean(dim=0)
                    if self.running_prob is None:
                        self.running_prob = batch_mean
                    else:
                        self.running_prob = 0.9 * self.running_prob + 0.1 * batch_mean

        assert probs is not None
        return probs[:, 1].detach()


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
TTA_REGISTRY.register("tent", TENTMethod)
TTA_REGISTRY.register("sar", SARMethod)
TTA_REGISTRY.register("eata", EATAMethod)


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
    method_kwargs: Optional[Dict[str, Any]] = None,
):
    optimizer = torch.optim.SGD
    prediction = []
    label = []
    method_kwargs = method_kwargs or {}

    registry_kwargs: Dict[str, Any] = {
        "model": model,
        "optimizer_type": optimizer,
        "device": device,
    }
    if method_name == "ftta":
        source_y = get_source_prior(exp, device)
        registry_kwargs["prior"] = source_y
        registry_kwargs["lr_list"] = [1e-5, 5e-4, 1e-4]

    if method_name == "tent":
        registry_kwargs.update(
            {
                "tent_lr": method_kwargs.get("tent_lr", 1e-3),
                "tent_steps": method_kwargs.get("tent_steps", 1),
                "tent_eps": method_kwargs.get("tent_eps", 1e-8),
            }
        )
    if method_name == "sar":
        registry_kwargs.update(
            {
                "sar_lr": method_kwargs.get("sar_lr", 1e-3),
                "sar_steps": method_kwargs.get("sar_steps", 1),
                "sar_rho": method_kwargs.get("sar_rho", 0.05),
                "sar_entropy_margin": method_kwargs.get("sar_entropy_margin", 0.4),
                "tent_eps": method_kwargs.get("tent_eps", 1e-8),
            }
        )
    if method_name == "eata":
        registry_kwargs.update(
            {
                "eata_lr": method_kwargs.get("eata_lr", 1e-3),
                "eata_steps": method_kwargs.get("eata_steps", 1),
                "eata_entropy_margin": method_kwargs.get("eata_entropy_margin", 0.4),
                "eata_diversity_margin": method_kwargs.get("eata_diversity_margin", 0.05),
                "tent_eps": method_kwargs.get("tent_eps", 1e-8),
            }
        )
    tta_method = TTA_REGISTRY.create(method_name, **registry_kwargs)

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
    method_kwargs: Optional[Dict[str, Any]] = None,
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
                method_kwargs=method_kwargs,
            )
            pre = np.round(pre)
            tta_score = sklearn.metrics.accuracy_score(tar, pre)
            print("\n", f"{method_name.upper()}: acc ", "\n", tta_score, "\n")

        with torch.no_grad():
            model.eval()
            pre, tar = get_predictions_and_labels_unadapt(model, loader, device)
            pre = np.round(pre)
            unadapt_score = sklearn.metrics.accuracy_score(tar, pre)
            print("\n", "Unadapt: acc ", "\n", unadapt_score, "\n")
            model.last_unadapt_ood_score = float(unadapt_score)
            model.last_tta_ood_score = float(tta_score)
    else:
        logging.info(f"TTA testing, only ood score will be display, split:{split} skipping")
        return 0

    return tta_score
