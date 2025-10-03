import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from scipy import sparse
from sklearn.model_selection import train_test_split
import random

# random seed
seed = 42
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

# load Karate Club graph
G = nx.karate_club_graph()
A = nx.adjacency_matrix(G)  # sparse adjacency (scipy)
N = G.number_of_nodes()

# using one-hot as X
use_one_hot = True
if use_one_hot:
    X = np.eye(N, dtype=np.float32)
else:
    X = np.array([d for _, d in G.degree()], dtype=np.float32).reshape(-1,1)

# labelling 'club'（'Mr. Hi' / 'Officer'）into 0/1
club = [G.nodes[i]['club'] for i in range(N)]
labels = np.array([0 if c == 'Mr. Hi' else 1 for c in club], dtype=np.int64)

# train/val/test set
idx = np.arange(N)
idx_train, idx_test, y_train, y_test = train_test_split(idx, labels, test_size=0.4, stratify=labels, random_state=seed)
idx_train, idx_val, _, _ = train_test_split(idx_train, labels[idx_train], test_size=0.25, stratify=labels[idx_train], random_state=seed)
# idx_train ~ 0.45N, idx_val ~ 0.15N, idx_test ~ 0.4N

# normalization \hat{A} = D^{-1/2} (A + I) D^{-1/2}
def normalize_adj(adj):
    adj = adj + sparse.eye(adj.shape[0])
    deg = np.array(adj.sum(1)).flatten()
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.
    D_inv_sqrt = sparse.diags(deg_inv_sqrt)
    return D_inv_sqrt @ adj @ D_inv_sqrt  # still sparse

A_norm = normalize_adj(A)
A_norm = torch.FloatTensor(A_norm.toarray())  # 转为 dense tensor（小图可行）

# torch tensor
X = torch.FloatTensor(X)
y = torch.LongTensor(labels)

# GCN layer（Kipf & Welling）
class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_feats, out_feats, bias=bias)

    def forward(self, X, A_norm):
        # X: [N, F], A_norm: [N, N]
        return A_norm @ self.linear(X)

class GCN(nn.Module):
    def __init__(self, in_feats, hidden, n_classes, dropout=0.5):
        super().__init__()
        self.gcn1 = GCNLayer(in_feats, hidden)
        self.gcn2 = GCNLayer(hidden, n_classes)
        self.dropout = dropout

    def forward(self, X, A_norm):
        h = self.gcn1(X, A_norm)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.gcn2(h, A_norm)
        return h

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(X.shape[1], 16, 2, dropout=0.5).to(device)
X = X.to(device); A_norm = A_norm.to(device); y = y.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# mask helper
train_mask = torch.zeros(N, dtype=torch.bool); train_mask[idx_train] = True
val_mask = torch.zeros(N, dtype=torch.bool); val_mask[idx_val] = True
test_mask = torch.zeros(N, dtype=torch.bool); test_mask[idx_test] = True
train_mask = train_mask.to(device); val_mask = val_mask.to(device); test_mask = test_mask.to(device)

def train(epoch):
    model.train()
    optimizer.zero_grad()
    out = model(X, A_norm)
    loss = F.cross_entropy(out[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate(mask):
    model.eval()
    logits = model(X, A_norm)
    preds = logits.argmax(dim=1)
    acc = (preds[mask] == y[mask]).float().mean().item()
    return acc

for epoch in range(1, 201):
    loss = train(epoch)
    if epoch % 10 == 0:
        acc_train = evaluate(train_mask)
        acc_val = evaluate(val_mask)
        acc_test = evaluate(test_mask)
        print(f"Epoch {epoch:03d}  Loss: {loss:.4f}  Train: {acc_train:.3f}  Val: {acc_val:.3f}  Test: {acc_test:.3f}")

# print acc after train
acc_test = evaluate(test_mask)
print("Final Test Accuracy:", acc_test)

# visualizaton
import matplotlib.pyplot as plt

@torch.no_grad()
def visualize():
    model.eval()
    logits = model(X, A_norm)
    pred = logits.argmax(dim=1).cpu().numpy()
    true = y.cpu().numpy()

    # spring_layout
    pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(12, 5))

    # --- left：truth ---
    plt.subplot(1, 2, 1)
    nx.draw(
        G, pos,
        node_color=true, cmap=plt.cm.Set1,
        with_labels=True, node_size=500, font_color="white"
    )
    plt.title("Ground Truth")

    # --- right：predict ---
    plt.subplot(1, 2, 2)
    nx.draw(
        G, pos,
        node_color=pred, cmap=plt.cm.Set1,
        with_labels=True, node_size=500, font_color="white"
    )
    plt.title("GCN Prediction")

    plt.savefig("res.png")

visualize()