import torch
import wandb
from arg_parser import load_dataset, load_loss_function, load_model
from result_parser import parse_sweep_results, save_json
from train import train_test_split, train
import numpy as np

sweep_configuration = {
    "method": "grid",
    "name": "num epochs gat-2l mse",
    "metric": {"goal": "minimize", "name": "test_loss"},
    "parameters": {
        "model": {"values": ["gat-2l"]},
        "loss": {"values": ["mse"]},
        "dataset": {"values": ["cora"]},
        "seed": {"min": 1, "max": 10}
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="Double Descent in GNNs")


def main():
    wandb.init()
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    graph, num_classes = load_dataset(wandb.config.dataset)
    eval_loss = load_loss_function(wandb.config.loss)
    # c_epochs = [int(x) for x in np.geomspace(2, 131068, num=17)]
    c_epochs = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    graph.to(device)

    for c in c_epochs:
        model = load_model(wandb.config.model, graph, num_classes, device)
        train_test_split(graph, num_classes, 0.4)
        test_loss, test_acc, train_loss = train(model, graph, eval_loss, c)

        print(c)
        wandb.log(
            {"num_epochs": c, "acc": test_acc, "test_loss": test_loss, "train_loss": train_loss}
        )

    wandb.finish()
    torch.cuda.empty_cache()


wandb.agent(sweep_id, function=main)
print(sweep_id)
parsed_results = parse_sweep_results(sweep_id, "num_epochs")
save_json(parsed_results, "run_data/num_epochs/num_epochs_gat_2l_mse.json")
