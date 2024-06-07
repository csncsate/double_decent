import numpy as np
import torch
from result_parser import parse_sweep_results, save_json
from train import train_test_split, train
import wandb
from arg_parser import load_model, load_dataset, load_loss_function

# Define sweep config
sweep_configuration = {
    "method": "grid",
    "name": "training_ratio - ensemble - csbm - l=0,g=3,m=1,e=1k",
    "metric": {"goal": "minimize", "name": "test_loss"},
    "parameters": {
        "dataset": {"values": ["csbm"]},
        "model": {"values": ["gcn-1l", "gcn-2l", "gcn-2l-do", "gat-1l", "gat-2l"]},
        "loss": {"values": ["mse", "cross-entropy"]},
        "seed": {"min": 1, "max": 2},
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="Double Descent in GNNs")


def main():
    wandb.init()
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    training_ratios = np.logspace(np.log2(0.01), np.log2(0.9), num=15, base=2)
    # training_ratios = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]

    graph, num_classes = load_dataset(wandb.config.dataset, graph_snr=0, inverse_sampling_ratio=3, node_feature_snr=1)
    eval_loss = load_loss_function(wandb.config.loss)

    for training_ratio in training_ratios:
        graph.to(device)
        model = load_model(wandb.config.model, graph, num_classes, device)
        train_test_split(graph, num_classes, training_ratio)
        test_loss, test_acc, train_loss = train(model, graph, eval_loss)

        wandb.log(
            {"training_ratio": training_ratio, "acc": test_acc, "test_loss": test_loss, "train_loss": train_loss})
        print(training_ratio)
    wandb.finish()
    torch.cuda.empty_cache()


wandb.agent(sweep_id, function=main)
print(sweep_id)
parsed_results = parse_sweep_results(sweep_id, "training_ratio")
save_json(parsed_results, f"run_data/{sweep_configuration.get('name')}.json")
