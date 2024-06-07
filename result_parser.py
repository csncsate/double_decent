import ast
import wandb
import numpy as np
from collections import defaultdict
import json


def aggregate_sweeps():
    res = defaultdict(dict)
    accs = defaultdict(list)
    test_losses = defaultdict(list)
    train_losses = defaultdict(list)
    for i in range(10):
        config_res = load_result_json(f'num_neurons_gat_ce_{i + 1}.json').get(("gat-2l", "cross-entropy", "cora"))
        for neuron_count, metrics in config_res.items():
            accs[neuron_count].append(metrics['acc_mean'])
            test_losses[neuron_count].append(metrics['test_loss_mean'])
            train_losses[neuron_count].append(metrics['train_loss_mean'])

    for neuron_count in accs:
        res[neuron_count] = {
            "acc_mean": np.mean(accs.get(neuron_count)),
            "acc_std": np.std(accs.get(neuron_count)),
            "test_loss_mean": np.mean(test_losses.get(neuron_count)),
            "test_loss_std": np.std(test_losses.get(neuron_count)),
            "train_loss_mean": np.mean(train_losses.get(neuron_count)),
            "train_loss_std": np.std(train_losses.get(neuron_count))
        }

    return res


def parse_sweep_results(sweep_id, param):
    api = wandb.Api()
    sweep = api.sweep(sweep_id)

    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for run in sweep.runs:
        # Fetch the configuration details
        model = run.config['model']
        loss_function = run.config['loss']
        dataset = run.config['dataset']

        # Fetch the historical data for each run
        history = run.history()

        # Iterate over the history data and aggregate by training_ratio
        for _, row in history.iterrows():
            test_param = row[param]
            results[(model, loss_function, dataset)][test_param]['acc'].append(row['acc'])
            results[(model, loss_function, dataset)][test_param]['test_loss'].append(row['test_loss'])
            results[(model, loss_function, dataset)][test_param]['train_loss'].append(row['train_loss'])

    aggregated_stats = defaultdict(lambda: defaultdict(dict))

    for config, ratios in results.items():
        for ratio, metrics in ratios.items():
            acc_mean = np.mean(metrics['acc'])
            acc_std = np.std(metrics['acc'])
            test_loss_mean = np.mean(metrics['test_loss'])
            test_loss_std = np.std(metrics['test_loss'])
            train_loss_mean = np.mean(metrics['train_loss'])
            train_loss_std = np.std(metrics['train_loss'])

            aggregated_stats[config][ratio] = {
                'acc_mean': acc_mean,
                'acc_std': acc_std,
                'test_loss_mean': test_loss_mean,
                'test_loss_std': test_loss_std,
                'train_loss_mean': train_loss_mean,
                'train_loss_std': train_loss_std
            }

    return aggregated_stats


def get_config_results(stats_dict, model, loss, dataset):
    selected_config = (model, loss, dataset)
    return stats_dict.get(selected_config)


def save_json(datadict, filename):
    data = {str(key): value for key, value in datadict.items()}
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


def load_result_json(filename):
    with open(filename, 'r') as f:
        data_loaded = json.load(f)
    return defaultdict(dict, {ast.literal_eval(key): value for key, value in data_loaded.items()})

    # aggregated_stats = parse_sweep_results("/Double Descent in GNNs/60b1eqk7", "training_ratio")
    # num_rows = len(aggregated_stats) // 4
    # num_cols = 4
    #
    # # Create a figure with subplots
    # fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10), sharex=True, sharey='row')
    #
    # # Flatten the axes array for easy iteration
    # axes = axes.flatten()
    #
    # # Iterate over each configuration and plot on a separate subplot
    # for i, (config, ratios_stats) in enumerate(aggregated_stats.items()):
    #     create_plot_for_config(config, ratios_stats, "Training ratio", axes[i])
    #
    # # Define custom handles for the shared legend
    # test_loss_handle = mlines.Line2D([], [], color='red', marker='o', label='Test Loss')
    # accuracy_handle = mlines.Line2D([], [], color='black', marker='o', label='Accuracy')
    # train_loss_handle = mlines.Line2D([], [], color='blue', marker='o', label='Train Loss')
    #
    # # Place a shared legend above the subplots
    # fig.legend(handles=[test_loss_handle, accuracy_handle, train_loss_handle], loc='upper center', ncol=3, frameon=False)
    #
    # # Adjust the layout
    # fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.show()
