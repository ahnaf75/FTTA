import argparse

import torch

from tableshift import get_dataset
from tableshift.FTTA_src import FederatedFTTAConfig, FederatedFTTAServer, partition_tensor_for_clients
from tableshift.models.default_hparams import get_default_config
from tableshift.models.utils import get_estimator


def main(experiment: str, cache_dir: str, model: str, num_clients: int, algorithm: str):
    dset = get_dataset(experiment, cache_dir)
    x_train, _, _, _ = dset.get_pandas("train")

    x_tensor = torch.tensor(x_train.values, dtype=torch.float32)
    client_data = partition_tensor_for_clients(x_tensor, num_clients=num_clients, shuffle=True)

    estimator = get_estimator(model, **get_default_config(model, dset))
    prior = torch.tensor([0.5, 0.5], dtype=torch.float32)

    config = FederatedFTTAConfig(
        algorithm=algorithm,
        num_clients=num_clients,
        rounds=3,
        local_steps=1,
        batch_size=128,
    )
    server = FederatedFTTAServer(
        model=estimator,
        optimizer_type=torch.optim.Adam,
        prior=prior,
        config=config,
    )
    server.fit(client_data)

    preds = server.predict(x_tensor[:10])
    print(f"{algorithm} federated FTTA completed for {num_clients} clients.")
    print(preds.squeeze())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="diabetes_readmission")
    parser.add_argument("--cache_dir", default="tmp")
    parser.add_argument("--model", default="mlp")
    parser.add_argument("--algorithm", default="fedavg", choices=["fedavg", "fedprox", "pfedgraph", "fedamp"])
    parser.add_argument("--num_clients", type=int, default=5)
    args = parser.parse_args()

    main(**vars(args))
