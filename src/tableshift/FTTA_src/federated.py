from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn.functional as F

from .FTTA import FTTA


def _clone_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in state_dict.items()}


def _average_state_dicts(
    state_dicts: List[Dict[str, torch.Tensor]],
    weights: Optional[List[float]] = None,
) -> Dict[str, torch.Tensor]:
    if not state_dicts:
        raise ValueError("state_dicts must be non-empty")
    if weights is None:
        weights = [1.0 / len(state_dicts)] * len(state_dicts)

    total = float(sum(weights))
    normalized = [w / total for w in weights]
    keys = state_dicts[0].keys()
    avg = {}
    for key in keys:
        avg[key] = sum(sd[key] * w for sd, w in zip(state_dicts, normalized))
    return avg


def _state_to_vector(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    flat = [v.detach().flatten().float().cpu() for v in state_dict.values()]
    return torch.cat(flat)


def partition_tensor_for_clients(
    x: torch.Tensor,
    num_clients: int,
    shuffle: bool = True,
) -> Dict[int, torch.Tensor]:
    if num_clients <= 0:
        raise ValueError("num_clients must be > 0")
    n = x.shape[0]
    order = torch.randperm(n) if shuffle else torch.arange(n)
    splits = torch.chunk(order, num_clients)
    return {cid: x[idx] for cid, idx in enumerate(splits)}


@dataclass
class FederatedFTTAConfig:
    algorithm: str = "fedavg"
    num_clients: int = 5
    rounds: int = 5
    local_steps: int = 1
    batch_size: int = 64
    mu: float = 1e-3
    fedamp_alpha: float = 5.0
    graph_topk: int = 2
    graph_temperature: float = 1.0


class FederatedFTTAServer:
    SUPPORTED_ALGORITHMS = {"fedavg", "fedprox", "pfedgraph", "fedamp"}

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer_type,
        prior: torch.Tensor,
        config: Optional[FederatedFTTAConfig] = None,
        device: Optional[str] = None,
    ):
        self.config = config or FederatedFTTAConfig()
        self.algorithm = self.config.algorithm.lower()
        if self.algorithm not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"algorithm must be one of {sorted(self.SUPPORTED_ALGORITHMS)}; got {self.algorithm}"
            )
        if self.config.num_clients <= 0:
            raise ValueError("num_clients must be > 0")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer_type = optimizer_type
        self.prior = prior.to(self.device)

        self.global_model = deepcopy(model).to(self.device)
        base_state = _clone_state_dict(self.global_model.state_dict())
        self.client_states: Dict[int, Dict[str, torch.Tensor]] = {
            cid: _clone_state_dict(base_state) for cid in range(self.config.num_clients)
        }

    def _build_client_ftta(self, state_dict: Dict[str, torch.Tensor]) -> FTTA:
        local_model = deepcopy(self.global_model).to(self.device)
        local_model.load_state_dict(state_dict)
        return FTTA(
            model=local_model,
            optimizer_type=self.optimizer_type,
            prior=self.prior.clone(),
            device=self.device,
        )

    def _mean_ftta_state(self, ftta_model: FTTA) -> Dict[str, torch.Tensor]:
        model_states = [m.state_dict() for m in ftta_model.model_list]
        return _average_state_dicts(model_states)

    def _apply_proximal_shrink(
        self,
        ftta_model: FTTA,
        reference_state: Dict[str, torch.Tensor],
        mu: float,
    ) -> None:
        if mu <= 0:
            return
        with torch.no_grad():
            for local_model in ftta_model.model_list:
                for name, param in local_model.named_parameters():
                    param.sub_(mu * (param - reference_state[name].to(param.device)))

    def _local_adapt(
        self,
        state_dict: Dict[str, torch.Tensor],
        client_data: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        ftta_model = self._build_client_ftta(state_dict)
        ftta_model.train()

        if client_data.ndim < 2:
            raise ValueError("client_data must be a batched feature tensor")

        batches = torch.split(client_data.to(self.device), self.config.batch_size)
        for _ in range(self.config.local_steps):
            for batch in batches:
                _ = ftta_model(batch)
                if self.algorithm == "fedprox":
                    self._apply_proximal_shrink(ftta_model, state_dict, mu=self.config.mu)

        return self._mean_ftta_state(ftta_model)

    def _pairwise_similarity(
        self,
        client_updates: Dict[int, Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        ids = sorted(client_updates)
        vectors = torch.stack([_state_to_vector(client_updates[i]) for i in ids])
        vectors = F.normalize(vectors, p=2, dim=1)
        return vectors @ vectors.T

    def _aggregate_fedavg(
        self,
        client_updates: Dict[int, Dict[str, torch.Tensor]],
    ) -> None:
        avg_state = _average_state_dicts([client_updates[cid] for cid in sorted(client_updates)])
        for cid in self.client_states:
            self.client_states[cid] = _clone_state_dict(avg_state)
        self.global_model.load_state_dict(avg_state)

    def _aggregate_personalized(
        self,
        client_updates: Dict[int, Dict[str, torch.Tensor]],
    ) -> None:
        ids = sorted(client_updates)
        sim = self._pairwise_similarity(client_updates)

        if self.algorithm == "fedamp":
            weights = F.softmax(self.config.fedamp_alpha * sim, dim=1)
        else:  # pfedgraph
            k = min(self.config.graph_topk, len(ids))
            mask = torch.zeros_like(sim)
            topk_vals, topk_idx = torch.topk(sim, k=k, dim=1)
            mask.scatter_(1, topk_idx, topk_vals)
            weights = F.softmax(mask / max(self.config.graph_temperature, 1e-6), dim=1)

        stacked = [client_updates[cid] for cid in ids]
        new_states: Dict[int, Dict[str, torch.Tensor]] = {}
        for row_idx, cid in enumerate(ids):
            row_w = weights[row_idx].tolist()
            new_states[cid] = _average_state_dicts(stacked, row_w)

        self.client_states.update(new_states)
        global_state = _average_state_dicts([self.client_states[cid] for cid in ids])
        self.global_model.load_state_dict(global_state)

    def fit(self, client_tensors: Dict[int, torch.Tensor]) -> None:
        if len(client_tensors) != self.config.num_clients:
            raise ValueError(
                f"expected exactly {self.config.num_clients} clients, got {len(client_tensors)}"
            )

        for _ in range(self.config.rounds):
            updates: Dict[int, Dict[str, torch.Tensor]] = {}
            for cid, data in client_tensors.items():
                updates[cid] = self._local_adapt(self.client_states[cid], data)

            if self.algorithm in {"fedavg", "fedprox"}:
                self._aggregate_fedavg(updates)
            else:
                self._aggregate_personalized(updates)

    @torch.no_grad()
    def predict(self, x: torch.Tensor, client_id: Optional[int] = None) -> torch.Tensor:
        if client_id is None:
            model = self.global_model
        else:
            model = deepcopy(self.global_model).to(self.device)
            model.load_state_dict(self.client_states[client_id])

        logits = model(x.to(self.device))
        return torch.sigmoid(logits).detach().cpu()
