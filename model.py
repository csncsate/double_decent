import torch
from torch_geometric.nn import GCNConv, GATConv
import torch.nn.functional as F


# Defining the Graph Convolutional Network
class GCNTwoLayerReLu(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_hidden_channels, dropout=False):
        super().__init__()
        self.conv1 = GCNConv(num_features, num_hidden_channels)
        self.conv2 = GCNConv(num_hidden_channels, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        if self.dropout:
            x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GCNSimple(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return x


class GATSimple(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.gat1 = GATConv(num_features, num_classes)

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        return x


class GATTwoLayer(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_hidden_channels):
        super().__init__()
        heads = 8
        self.gat1 = GATConv(num_features, num_hidden_channels, heads)
        self.gat2 = GATConv(num_hidden_channels * heads, num_classes)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.gat1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.gat2(x, edge_index)
        return x
