from CONSTANTS import *
from torch_geometric.nn import HypergraphConv, global_mean_pool, global_add_pool, global_max_pool
import torch.nn as nn
import torch.nn.functional as F
from util import logger


class AttentionPooling(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attn = nn.Linear(in_dim, 1)

    def forward(self, x, batch=None):
        attn_weights = F.softmax(self.attn(x), dim=0)
        return torch.sum(attn_weights * x, dim=0)


class HypergraphEncoder(nn.Module):
    def __init__(self, in_dim=300, hidden_dim=256, out_dim=128, dropout=0.5, pooling='attention'):
        super().__init__()
        self.dropout = dropout
        self.conv1 = HypergraphConv(in_dim, hidden_dim)
        self.conv2 = HypergraphConv(hidden_dim, hidden_dim)
        self.conv3 = HypergraphConv(hidden_dim, out_dim)
        if pooling == 'mean':
            self.pooling = global_mean_pool
        elif pooling == 'add':
            self.pooling = global_add_pool
        elif pooling == 'max':
            self.pooling = global_max_pool
        elif pooling == 'attention':
            self.pooling = AttentionPooling(out_dim)

    def forward(self, x, hyper_edge_index):
        x = F.leaky_relu(self.conv1(x, hyper_edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.leaky_relu(self.conv2(x, hyper_edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, hyper_edge_index)

        return self.pooling(x)


class HyperGraphDeepSVDD(nn.Module):
    def __init__(self, encoder, center=None):
        super().__init__()
        self.encoder = encoder
        self.center = center

    def init_center(self, train_data, template_embedding, eps=0.1):
        self.encoder.eval()
        z = []
        with torch.no_grad():
            for data in train_data:
                if data.y == 0:
                    out = self.encoder(template_embedding.get_embeddings(data.x), data.hyper_edge_index)
                    z.append(out.detach())
        self.center = torch.mean(torch.cat(z, dim=0), dim=0)
        self.center = ((self.center / (self.center.norm() + 1e-8)) * eps)

    def forward(self, x, hyper_edge_index):
        z = self.encoder(x, hyper_edge_index)
        return torch.sum((z - self.center) ** 2)


class HybridSVDDLoss(nn.Module):
    def __init__(self, nu=0.1, alpha=0.5):
        super().__init__()
        self.nu = nu
        self.alpha = alpha

    def forward(self, distances, labels=None):
        unsup_loss = torch.mean(distances)

        if labels is not None:
            sup_loss = torch.mean(
                torch.where(labels == 0,
                            distances,
                            torch.relu(1 - distances.sqrt()))
            )
            total_loss = (1 - self.alpha) * unsup_loss + self.alpha * sup_loss
        else:
            total_loss = unsup_loss

        return total_loss

def compute_threshold(model, datas, template_embedding, quantile=0.95):
    model.eval()
    distances = []
    with torch.no_grad():
        for data in datas:
            if data.y == 0:
                dist = model(template_embedding.get_embeddings(data.x), data.hyper_edge_index).unsqueeze(0)
                distances.append(dist.cpu())
    return np.quantile(torch.cat(distances).numpy(), quantile).item()


def predict_anomaly_score(model, data):
    with torch.no_grad():
        distance = model(data.x, data.hyper_edge_index)
    return distance.item()
