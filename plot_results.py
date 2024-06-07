# def plot_results(all_test_losses, all_train_losses, all_test_accuracies, test_param, log, test_param_name):
#     # Convert lists to numpy arrays for easy computation
#     all_test_accuracies = np.array(all_test_accuracies)
#     all_test_losses = np.array(all_test_losses)
#     all_train_losses = np.array(all_train_losses)
#
#     # Calculate mean and standard deviation
#     mean_test_accuracies = np.mean(all_test_accuracies, axis=0)
#     std_test_accuracies = np.std(all_test_accuracies, axis=0)
#
#     mean_test_losses = np.mean(all_test_losses, axis=0)
#     std_test_losses = np.std(all_test_losses, axis=0)
#
#     mean_train_losses = np.mean(all_train_losses, axis=0)
#     std_train_losses = np.std(all_train_losses, axis=0)
#
#     # Plotting the results
#     fig, ax1 = plt.subplots()
#
#     # Plotting mean test loss on the left y-axis (red)
#     loss_color = 'red'
#     ax1.set_xlabel(test_param_name)
#     ax1.set_ylabel('Test Loss', color=loss_color)
#     ax1.plot(test_param, mean_test_losses, label='Test Loss', marker='o', color=loss_color)
#     ax1.tick_params(axis='y', labelcolor=loss_color)
#     ax1.scatter(test_param, mean_test_losses, color=loss_color)
#
#     # Set logarithmic scale for x-axis with base 2
#     if log:
#         ax1.set_xscale('log', base=2)
#     ax1.set_xticks(test_param)
#     ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
#     ax1.set_xlim(left=test_param[0] * 0.8)
#
#     # Creating a secondary y-axis for mean test accuracy on the right side (black)
#     ax2 = ax1.twinx()
#     acc_color = 'black'
#     ax2.set_ylabel('Test Accuracy', color=acc_color)
#     ax2.plot(test_param, mean_test_accuracies, label='Test Accuracy', marker='o', color=acc_color)
#     ax2.tick_params(axis='y', labelcolor=acc_color)
#     ax2.scatter(test_param, mean_test_accuracies, color=acc_color)
#
#     ax3 = ax1.twinx()
#     train_loss_color = 'blue'
#     ax3.set_ylabel('Train Loss', color=train_loss_color)
#     ax3.plot(test_param, mean_train_losses, label='Train Loss', marker='o', color=train_loss_color)
#     ax3.tick_params(axis='y', labelcolor=train_loss_color)
#     ax3.scatter(test_param, mean_train_losses, color=train_loss_color)
#
#     # Plotting standard deviation as vertical lines
#     ax1.vlines(test_param, mean_test_losses - std_test_losses, mean_test_losses + std_test_losses,
#                color=loss_color, alpha=0.5)
#     ax2.vlines(test_param, mean_test_accuracies - std_test_accuracies, mean_test_accuracies + std_test_accuracies,
#                color=acc_color, alpha=0.5)
#     ax3.vlines(test_param, mean_train_losses - std_train_losses, mean_train_losses + std_train_losses,
#                color=train_loss_color, alpha=0.5)
#
#     fig.tight_layout()
#     plt.title('Test Loss and Accuracy')
#     plt.show()
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import ticker

from result_parser import load_result_json, get_config_results


def create_plot_for_metric(test_param_steps, means, stds, ax, color):
    ax.plot(test_param_steps, means, marker='o', color=color)
    ax.fill_between(test_param_steps, means - stds, means + stds, color=color, alpha=0.2)
    ax.tick_params(axis='y', labelcolor=color)


def create_plot_for_config(
        config_results,
        test_param_name,
        acc_ax,
        train_ax,
        test_ax,
        show_xlabels=True,
        show_ylabels_left=True,
        show_ylabels_right=True
):
    param_steps = []
    mean_test_losses = []
    std_test_losses = []
    mean_accuracies = []
    std_accuracies = []
    mean_train_losses = []
    std_train_losses = []

    for current_step in config_results.keys():
        stats = config_results[current_step]
        param_steps.append(current_step)
        mean_test_losses.append(stats['test_loss_mean'])
        std_test_losses.append(stats['test_loss_std'])
        mean_accuracies.append(stats['acc_mean'])
        std_accuracies.append(stats['acc_std'])
        mean_train_losses.append(stats['train_loss_mean'])
        std_train_losses.append(stats['train_loss_std'])

    param_steps = np.array(param_steps, dtype=float)
    mean_test_losses = np.array(mean_test_losses, dtype=float)
    std_test_losses = np.array(std_test_losses, dtype=float)
    mean_accuracies = np.array(mean_accuracies, dtype=float)
    std_accuracies = np.array(std_accuracies, dtype=float)
    mean_train_losses = np.array(mean_train_losses, dtype=float)
    std_train_losses = np.array(std_train_losses, dtype=float)

    acc_ax.set_xscale('log', base=2)
    acc_ax.xaxis.set_major_locator(ticker.LogLocator(base=2.0, numticks=6))

    # Create individual plots for each metric
    create_plot_for_metric(param_steps, mean_accuracies, std_accuracies, acc_ax, 'black')
    create_plot_for_metric(param_steps, mean_train_losses, std_train_losses, train_ax, 'blue')
    create_plot_for_metric(param_steps, mean_test_losses, std_test_losses, test_ax, 'red')

    if show_xlabels:
        acc_ax.set_xlabel(test_param_name)
        acc_ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    else:
        acc_ax.xaxis.set_major_formatter(ticker.NullFormatter())

    if not show_ylabels_left:
        acc_ax.yaxis.set_major_locator(ticker.NullLocator())
        train_ax.yaxis.set_major_locator(ticker.NullLocator())
    else:
        acc_ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        train_ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))

    if not show_ylabels_right:
        test_ax.yaxis.set_major_locator(ticker.NullLocator())
    else:
        test_ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))


def plot_config_results(filename, config, test_param_name):
    stats = load_result_json(filename)
    config_stats = get_config_results(stats, config[0], config[1], config[2])

    fig, ax = plt.subplots()
    create_plot_for_config(config_stats, test_param_name, ax, ax.twinx(), ax.twinx())

    test_loss_handle = mlines.Line2D([], [], color='red', marker='o', label='Test Loss')
    accuracy_handle = mlines.Line2D([], [], color='black', marker='o', label='Accuracy')
    train_loss_handle = mlines.Line2D([], [], color='blue', marker='o', label='Train Loss')

    fig.legend(handles=[test_loss_handle, accuracy_handle, train_loss_handle], loc='upper center', ncol=3,
               frameon=False)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_ensemble_configs(filename, config_matrix, test_param_name):
    # Load stats from JSON
    stats = load_result_json(filename)

    # Determine the size of the grid
    num_rows, num_cols = len(config_matrix), len(config_matrix[0])

    # Create a figure with adjusted subplots arranged in a grid
    # Increasing width of each column and overall figure height
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows),
                             layout="constrained")

    # Iterate over the grid and plot each config
    for i in range(num_rows):
        for j in range(num_cols):
            config = config_matrix[i][j]
            ax = axes[i, j] if num_rows > 1 and num_cols > 1 else (axes[i] if num_cols == 1 else axes[j])
            config_stats = get_config_results(stats, config[0], config[1], config[2])
            print(stats, config_stats)

            show_xlabels = False
            show_ylabels_left = False
            show_ylabels_right = False

            # Set column and row titles
            if i == 0:
                ax.set_title(config[1], fontsize=14)
            if j == 0:
                ax.set_ylabel(config[1], fontsize=14, labelpad=15)
                show_ylabels_left = True
            if i == num_rows - 1:
                show_xlabels = True
            if j == num_cols - 1:
                show_ylabels_right = True

            ax2 = ax.twinx()
            ax3 = ax2.twinx()
            create_plot_for_config(
                config_stats,
                test_param_name,
                ax,
                ax2,
                ax3,
                True,
                True,
                True
            )

    test_loss_handle = mlines.Line2D([], [], color='red', marker='o', label='Test Loss')
    accuracy_handle = mlines.Line2D([], [], color='black', marker='o', label='Accuracy')
    train_loss_handle = mlines.Line2D([], [], color='blue', marker='o', label='Train Loss')
    fig.legend(handles=[test_loss_handle, accuracy_handle, train_loss_handle], loc='outside upper center', ncol=3,
               frameon=False)

    plt.show()
