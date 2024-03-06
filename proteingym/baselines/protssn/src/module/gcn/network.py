import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


class GCN(torch.nn.Module):
    def __init__(self, config, feat_type, input_dim, out_dim):
        super(GCN, self).__init__()
        self.config = config
        self.feat_type = feat_type
        self.hidden_dim = config["hidden_channels"]
        self.input_dim, self.out_dim = input_dim, out_dim

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(input_dim, self.hidden_dim))
        
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(self.hidden_dim))
        for _ in range(config["n_layers"] - 2):
            self.convs.append(
                GCNConv(self.hidden_dim, self.hidden_dim))
            self.bns.append(torch.nn.BatchNorm1d(self.hidden_dim))
        self.convs.append(GCNConv(self.hidden_dim, self.out_dim))
        
        self.dropout_prob = config["dropout"]
        

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
        x, pos, mu_r_norm, edge_index, edge_attr, batch, = (
            data.x.float(),
            data.pos.float(),
            data.mu_r_norm.float(),
            data.edge_index,
            data.edge_attr.float(),
            data.batch)
        
        input_x = torch.empty([pos.shape[0], 0]).to(x.device)
        if "manual" in self.feat_type:
            input_x = torch.cat([input_x, x], dim=1)
        if "esm" in self.feat_type:
            esm_rep = data.esm_rep.float()
            input_x = torch.cat([input_x, esm_rep], dim=1)
        x = input_x
        edge_index = edge_index
        
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_prob, training=self.training)
        x = self.convs[-1](x,edge_index)
        return x