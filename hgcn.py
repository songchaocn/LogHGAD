from torch_geometric.nn import HypergraphConv, global_mean_pool, global_add_pool, global_max_pool
from CONSTANTS import *


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2, pos_weight=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight  

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none',
                                                      weight=torch.tensor([self.pos_weight]))
        pt = torch.exp(-BCE_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        weighted_loss = torch.where(targets == 1, loss * self.pos_weight, loss)
        return weighted_loss.mean()


class AttentionPooling(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attn = nn.Linear(in_dim, 1)

    def forward(self, x, batch=None):
        attn_weights = F.softmax(self.attn(x), dim=0)
        return torch.sum(attn_weights * x, dim=0), attn_weights


class LogHypergraphConvNetwork(torch.nn.Module):
    def __init__(self, in_channels=300, hidden_channels=512, out_channels=300, dropout=0.5,
                 pooling='mean'):
        super(LogHypergraphConvNetwork, self).__init__()
        self.dropout = dropout
        self.conv1 = HypergraphConv(in_channels, hidden_channels)
        self.conv2 = HypergraphConv(hidden_channels, out_channels)

        if pooling == 'mean':
            self.pooling = global_mean_pool
        elif pooling == 'add':
            self.pooling = global_add_pool
        elif pooling == 'max':
            self.pooling = global_max_pool
        elif pooling == 'attention':
            self.pooling = AttentionPooling(out_channels)
        else:
            raise ValueError("Invalid pooling type. Choose from 'mean', 'add', or 'max'.")
        self.mlp = nn.Sequential(
            nn.Linear(out_channels, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(32, 1)
        )
        

    def forward(self, x, hyperedge_index):
        raw_x = x
        x = self.conv1(x, hyperedge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, hyperedge_index) + raw_x
        x = F.leaky_relu(x)

        graph_embedding, attn_weights = self.pooling(x, batch=None)
        graph_embedding = F.dropout(graph_embedding, self.dropout, training=self.training)
        return self.mlp(graph_embedding), attn_weights
