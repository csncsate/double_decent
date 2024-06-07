import argparse
from torch_geometric.datasets import Planetoid, WikipediaNetwork
import csbm_generator as csbm
from model import GCNTwoLayerReLu, GCNSimple, GATTwoLayer, GATSimple
from train import eval_mse_loss, eval_cross_entropy_loss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('loss', type=str, choices=['cross-entropy', 'mse'])
    parser.add_argument('model', type=str, choices=['gcn-2l', 'gcn-1l', 'gat-2l'])
    parser.add_argument('dataset', type=str, choices=['cora', 'csbm'])
    return parser.parse_args()


def load_dataset(name, **kwargs):
    if name == 'cora':
        dataset = Planetoid(root='/tmp/Cora', name='Cora')
        graph = dataset[0]
        num_classes = dataset.num_classes
    elif name == 'citeseer':
        dataset = Planetoid(root='/tmp/CiteSeer', name='CiteSeer')
        graph = dataset[0]
        num_classes = dataset.num_classes
    elif name == 'chameleon':
        dataset = WikipediaNetwork(root='/tmp/chameleon', name='chameleon')
        graph = dataset[0]
        num_classes = dataset.num_classes
    elif name == 'squirrel':
        dataset = WikipediaNetwork(root='/tmp/squirrel', name='squirrel')
        graph = dataset[0]
        num_classes = dataset.num_classes
    elif name == 'csbm':
        default_params = {
            'graph_snr': 1,
            'node_feature_snr': 1,
            'num_nodes': 2500,
            'inverse_sampling_ratio': 5,
            'average_degree': 10,
            'is_directed': True
        }

        default_params.update(kwargs)
        graph = csbm.generate_csbm(
            graph_snr=default_params['graph_snr'],
            node_feature_snr=default_params['node_feature_snr'],
            num_nodes=default_params['num_nodes'],
            inverse_sampling_ratio=default_params['inverse_sampling_ratio'],
            average_degree=default_params['average_degree'],
            is_directed=default_params['is_directed']
        )
        num_classes = 2
    else:
        raise ValueError("Invalid dataset argument.")
    return graph, num_classes


def load_loss_function(name):
    if name == 'mse':
        eval_loss = eval_mse_loss
    elif name == 'cross-entropy':
        eval_loss = eval_cross_entropy_loss
    else:
        raise ValueError("Invalid loss function.")
    return eval_loss


def load_model(name, graph, num_classes, device, c_hidden_channels=16):
    if name == 'gcn-2l':
        model = GCNTwoLayerReLu(graph.num_node_features, num_classes, c_hidden_channels).to(device)
    elif name == 'gcn-1l':
        model = GCNSimple(graph.num_node_features, num_classes).to(device)
    elif name == 'gcn-2l-do':
        model = GCNTwoLayerReLu(graph.num_node_features, num_classes, c_hidden_channels, True).to(device)
    elif name == 'gat-2l':
        model = GATTwoLayer(graph.num_node_features, num_classes, c_hidden_channels).to(device)
    elif name == 'gat-1l':
        model = GATSimple(graph.num_node_features, num_classes).to(device)
    else:
        raise ValueError("Invalid model architecture.")
    return model
