import torch
import wandb
from arg_parser import load_dataset, load_loss_function, load_model
from result_parser import parse_sweep_results, save_json
from train import train_with_batches, train_test_split, train
import numpy as np

sweep_configuration = {
    "method": "grid",
    "name": "neuron count - csbm - gcn-2l - negative lambda test",
    "metric": {"goal": "minimize", "name": "test_loss"},
    "parameters": {
        "dataset": {"values": ["csbm"]},
        "model": {"values": ["gcn-2l"]},
        "loss": {"values": ["cross-entropy", "mse"]},
        "graph_snr": {"min": -2, "max": 2},
        "seed": {"min": 1, "max": 4}
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="Double Descent in GNNs")


def main():
    wandb.init()
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    graph, num_classes = load_dataset(wandb.config.dataset, graph_snr=wandb.config.graph_snr, inverse_sampling_ratio=1, node_feature_snr=1)
    eval_loss = load_loss_function(wandb.config.loss)
    # c_hidden_channels = [2, 4, 8, 12, 16, 24, 32, 64, 128, 256, 512, 1024, 1448, 2048, 2896, 4096, 5792, 8192, 11585, 16384]
    # c_hidden_channels = [4, 8, 12, 16, 24, 32, 64, 128, 256, 512, 1024, 1448, 2048, 2896, 4096, 5792, 8192]
    c_hidden_channels = [4, 8, 12, 16, 24, 32, 64, 128, 256, 512, 1024, 1448, 2048, 2896, 4096]
    # c_hidden_channels = [4, 8, 12, 16, 24, 32, 64, 128, 256, 512, 1024, 1448, 2048]
    graph.to(device)

    for c in c_hidden_channels:
        torch.manual_seed(wandb.config.seed + c * 1000 + 100)
        model = load_model(wandb.config.model, graph, num_classes, device, c, wandb.config.seed)
        train_test_split(graph, num_classes, 0.4)
        test_loss, test_acc, train_loss = train(model, graph, eval_loss, n_iterations=1000)

        print(c)
        wandb.log(
            {"num_neurons": c, "acc": test_acc, "test_loss": test_loss, "train_loss": train_loss}
        )

    wandb.finish()
    torch.cuda.empty_cache()


wandb.agent(sweep_id, function=main)
print(sweep_id)
parsed_results = parse_sweep_results(sweep_id, "num_neurons")
save_json(parsed_results, f"run_data/{sweep_configuration.get('name')}.json")
