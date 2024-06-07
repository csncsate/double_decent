import numpy as np
from torch_geometric.data import Data
import torch


def generate_csbm(graph_snr, node_feature_snr, num_nodes, inverse_sampling_ratio, average_degree, is_directed=False):
    """
    Generate a graph and features for the Contextual Stochastic Block Model (CSBM) and return
    it in a format compatible with PyTorch Geometric Data type.

    Parameters:
    graph_snr (float): Graph signal-to-noise ratio.
    node_feature_snr (float): Node features signal-to-noise ratio.
    num_nodes (int): Number of nodes in the graph.
    inverse_sampling_ratio (float): Inverse sampling ratio to scale the number of features.
    is_directed (bool): Whether the graph is directed.

    Returns:
    Data: PyTorch Geometric Data object with the generated graph and features.
    """

    # Scale the number of features according to gamma and num_nodes
    num_features = int(num_nodes / inverse_sampling_ratio)

    # Generate class labels
    class_labels = np.array([0 if i < num_nodes // 2 else 1 for i in range(num_nodes)])

    # Generate the adjacency matrix
    adjacency_matrix = generate_adjacency_matrix(average_degree, signal_noise_ratio=graph_snr,
                                                 num_nodes=num_nodes, class_labels=class_labels,
                                                 is_directed=is_directed)

    # Generate the node features
    node_features = generate_node_features(num_nodes=num_nodes, num_features=num_features,
                                           signal_noise_ratio=node_feature_snr, class_labels=class_labels)

    # Convert adjacency matrix to edge index format expected by PyTorch Geometric
    edge_index = np.vstack(np.nonzero(adjacency_matrix)).astype(np.int64)
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    # Convert node features to tensor
    node_features = torch.tensor(node_features, dtype=torch.float)

    # Create PyTorch Geometric Data object
    data = Data(x=node_features, edge_index=edge_index, y=torch.tensor(class_labels, dtype=torch.long), num_classes=2,
                is_directed=is_directed)

    train_mask = torch.full((num_nodes,), False, dtype=torch.bool)
    test_mask = torch.full((num_nodes,), False, dtype=torch.bool)

    data.train_mask = train_mask
    data.test_mask = test_mask

    return data


def generate_adjacency_matrix(avg_degree, signal_noise_ratio, num_nodes, class_labels, is_directed):
    """
    Generate an adjacency matrix for the Contextual Stochastic Block Model (CSBM).

    Parameters:
    avg_degree (float): Average degree.
    signal_noise_ratio (float): Controls the homophily level. Positive for homophilic, negative for heterophilic.
    num_nodes (int): The total number of nodes in the graph.
    is_directed (bool): Whether the graph is directed.

    Returns:
    np.array: The generated adjacency matrix.
    """

    # Calculate cin and cout from d and signal_noise_ratio
    cin = avg_degree + np.sqrt(avg_degree) * signal_noise_ratio
    cout = avg_degree - np.sqrt(avg_degree) * signal_noise_ratio

    # Initialize the adjacency matrix
    adjacency_matrix = np.zeros((num_nodes, num_nodes))

    # Fill in the adjacency matrix
    for i in range(num_nodes):
        for j in range(num_nodes if is_directed else i):
            if class_labels[i] == class_labels[j]:
                # Nodes belong to the same community
                if np.random.rand() < cin / num_nodes:
                    adjacency_matrix[i][j] = 1
                    if not is_directed:
                        adjacency_matrix[j][i] = 1
            else:
                # Nodes belong to different communities
                if np.random.rand() < cout / num_nodes:
                    adjacency_matrix[i][j] = 1
                    if not is_directed:
                        adjacency_matrix[j][i] = 1

    return adjacency_matrix


def generate_node_features(num_nodes, num_features, signal_noise_ratio, class_labels):
    """
    Generate node features for the Contextual Stochastic Block Model (CSBM) following the spiked covariance model.

    Parameters:
    num_nodes (int): The total number of nodes.
    num_features (int): The number of features per node.
    signal_noise_ratio (float): Controls the Signal-to-Noise Ratio (SNR) of the features.
    class_labels (np.array): The class labels for the nodes.

    Returns:
    np.array: The generated node features matrix.
    """

    # Hidden feature vector u, drawn from a Gaussian distribution with mean 0 and variance 1/F
    u = np.random.normal(0, 1 / np.sqrt(num_features), num_features)

    # Noise matrix Ξ, each element drawn from a Gaussian distribution with mean 0 and variance 1/F
    xi = np.random.normal(0, 1 / np.sqrt(num_features), (num_nodes, num_features))

    # Adjust class_labels to be -1 for the second class
    adjusted_class_labels = np.where(class_labels == 0, -1, class_labels)

    # Calculate the feature matrix X
    node_features = np.sqrt(signal_noise_ratio / num_nodes) * adjusted_class_labels[:, np.newaxis] * u + xi

    return node_features
