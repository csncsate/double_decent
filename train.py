import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader


def eval_accuracy(model, graph, mask):
    model.eval()
    with torch.no_grad():
        out = model(graph.x, graph.edge_index)
        pred = out[mask].argmax(dim=1)
        correct = pred == graph.y[mask]
        acc = correct.sum().item() / mask.sum().item()
    return acc


def eval_mse_loss(output, data, mask):
    return F.mse_loss(output[mask], F.one_hot(data.y[mask], output.size()[1]).float())


def eval_cross_entropy_loss(output, data, mask):
    return F.cross_entropy(output[mask], data.y[mask])


def train_test_split(graph, num_classes, training_ratio):
    train_mask = torch.zeros(graph.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(graph.num_nodes, dtype=torch.bool)

    y = graph.y.detach().clone().cpu()
    for c in range(num_classes):
        idx = (y == c).nonzero(as_tuple=False).view(-1)
        idx = idx[torch.randperm(idx.size(0))]
        nodes_per_class = int(idx.size(0) * training_ratio)
        idx = idx[:nodes_per_class]
        train_mask[idx] = True

    remaining = (~train_mask).nonzero(as_tuple=False).view(-1)
    test_mask[remaining] = True

    graph.train_mask = train_mask
    graph.test_mask = test_mask


def train_with_batches(model, data, eval_loss, n_iterations, device, batch_size=16):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.00001)
    test_loss = float('inf')
    train_loss = float('inf')
    test_acc = 0

    for i in range(n_iterations):
        model.train()
        train_loader = NeighborLoader(data, input_nodes=data.train_mask, num_neighbors=[-1], batch_size=batch_size,
                                      shuffle=True)
        test_loader = NeighborLoader(data, input_nodes=data.test_mask, num_neighbors=[-1], batch_size=batch_size,
                                     shuffle=True)
        total_loss = 0
        c_train_samples = 0
        for batch in train_loader:
            batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = eval_loss(out, batch, batch.train_mask)
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().clone().cpu().numpy().item() * batch.train_mask.sum()
            c_train_samples += batch.train_mask.sum()

        total_loss = total_loss / c_train_samples
        if total_loss < train_loss:
            train_loss = total_loss
            model.eval()
            total_test_loss = 0
            total_correct = 0
            c_test_samples = 0

            for test_batch in test_loader:
                test_batch.to(device)
                out = model(test_batch.x, test_batch.edge_index)
                loss = eval_loss(out, test_batch, test_batch.test_mask)
                total_test_loss += loss.detach().clone().cpu().numpy().item() * test_batch.test_mask.sum()
                total_correct += eval_accuracy(model, test_batch, test_batch.test_mask) * test_batch.test_mask.sum()
                c_test_samples += test_batch.test_mask.sum()

            test_acc = total_correct / c_test_samples
            test_loss = total_test_loss / c_test_samples
    return test_loss, test_acc, train_loss


def train(model, data, eval_loss, n_iterations=1000, weight_decay=0.00001):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=weight_decay)
    best_loss = float('inf')
    train_loss = float('inf')
    test_loss = float('inf')
    test_acc = 0

    for _ in range(n_iterations):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = eval_loss(out, data, data.train_mask)
        loss.backward()
        optimizer.step()
        t = loss.item()

        if t < best_loss:
            best_loss = t
            model.eval()
            out = model(data.x, data.edge_index)

            loss = eval_loss(out, data, data.test_mask)
            test_loss = loss.detach().clone().cpu().numpy().item()

            loss = eval_loss(out, data, data.train_mask)
            train_loss = loss.detach().clone().cpu().numpy().item()

            test_acc = eval_accuracy(model, data, data.test_mask)
    return test_loss, test_acc, train_loss
