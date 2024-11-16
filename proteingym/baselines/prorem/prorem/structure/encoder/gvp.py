import torch
import torch.nn as nn
from .layer import GVP, GVPConvLayer, LayerNorm
from torch_scatter import scatter_mean

class AttentionPooling(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(AttentionPooling, self).__init__()
        self.attention_dim = attention_dim
        self.query_layer = nn.Linear(input_dim, attention_dim, bias=True)
        self.key_layer = nn.Linear(input_dim, attention_dim, bias=True)
        self.value_layer = nn.Linear(input_dim, 1, bias=True)  # value layer outputs one score
        self.softmax = nn.Softmax(dim=1)

    def forward(self, nodes_features1, nodes_features2):
        # Assuming nodes_features1 and nodes_features2 are both of shape [node_num, 128]
        nodes_features = nodes_features1 + nodes_features2  # This can also be concatenation or another operation

        query = self.query_layer(nodes_features)
        key = self.key_layer(nodes_features)
        value = self.value_layer(nodes_features)

        attention_scores = torch.matmul(query, key.transpose(-2, -1))  # [node_num, node_num]
        attention_scores = self.softmax(attention_scores)

        pooled_features = torch.matmul(attention_scores, value)  # [node_num, 1]
        return pooled_features

class AutoGraphEncoder(nn.Module):
    def __init__(self, node_in_dim, node_h_dim, 
                 edge_in_dim, edge_h_dim, attention_dim=64,
                 num_layers=4, drop_rate=0.1) -> None:
        super().__init__()
        self.W_v = nn.Sequential(
            LayerNorm(node_in_dim),
            GVP(node_in_dim, node_h_dim, activations=(None, None))
        )
        self.W_e = nn.Sequential(
            LayerNorm(edge_in_dim),
            GVP(edge_in_dim, edge_h_dim, activations=(None, None))
        )
        
        self.layers = nn.ModuleList(
                GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate) 
            for _ in range(num_layers))
        
        ns, _ = node_h_dim
        self.W_out = nn.Sequential(
            LayerNorm(node_h_dim),
            GVP(node_h_dim, (ns, 0)))

        self.dense = nn.Sequential(
            nn.Linear(ns, 2*ns), 
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(2*ns, node_in_dim[0]) # label num
        )
        
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, h_V, edge_index, h_E, node_s_labels):
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        for layer in self.layers:
            h_V = layer(h_V, edge_index, h_E)
        out = self.W_out(h_V)
        logits = self.dense(out)
        loss = self.loss_fn(logits, node_s_labels)
        
        return loss, logits
    
    def get_embedding(self, h_V, edge_index, h_E):
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        for layer in self.layers:
            h_V = layer(h_V, edge_index, h_E)
        out = self.W_out(h_V)
        return out
        


class SubgraphClassficationModel(nn.Module):
    '''   
    :param node_in_dim: node dimensions in input graph, should be
                        (6, 3) if using original features
    :param node_h_dim: node dimensions to use in GVP-GNN layers
    :param edge_in_dim: edge dimensions in input graph, should be
                        (32, 1) if using original features
    :param edge_h_dim: edge dimensions to embed to before use
                       in GVP-GNN layers
    :param num_layers: number of GVP-GNN layers
    :param drop_rate: rate to use in all dropout layers
    '''
    def __init__(self, node_in_dim, node_h_dim, 
                 edge_in_dim, edge_h_dim, attention_dim=64,
                 num_layers=4, drop_rate=0.1):
        
        super(SubgraphClassficationModel, self).__init__()
        self.W_v = nn.Sequential(
            LayerNorm(node_in_dim),
            GVP(node_in_dim, node_h_dim, activations=(None, None))
        )
        self.W_e = nn.Sequential(
            LayerNorm(edge_in_dim),
            GVP(edge_in_dim, edge_h_dim, activations=(None, None))
        )
        
        self.layers = nn.ModuleList(
                GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate) 
            for _ in range(num_layers))
        
        ns, _ = node_h_dim
        self.W_out = nn.Sequential(
            LayerNorm(node_h_dim),
            GVP(node_h_dim, (ns, 0)))
        
        self.attention_classifier = AttentionPooling(ns, attention_dim)
        self.dense = nn.Sequential(
            nn.Linear(2*ns, 2*ns), 
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(2*ns, 1)
        )
        
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, h_V_parent, edge_index_parent, h_E_parent, batch_parent,
                h_V_subgraph, edge_index_subgraph, h_E_subgraph, batch_subgraph,
                labels):      
        '''
        :param h_V: tuple (s, V) of node embeddings
        :param edge_index: `torch.Tensor` of shape [2, num_edges]
        :param h_E: tuple (s, V) of edge embeddings
        '''
        h_V_parent = self.W_v(h_V_parent)
        h_E_parent = self.W_e(h_E_parent)
        for layer in self.layers:
            h_V_parent = layer(h_V_parent, edge_index_parent, h_E_parent)
        out_parent = self.W_out(h_V_parent)
        out_parent = scatter_mean(out_parent, batch_parent, dim=0)
        
        h_V_subgraph = self.W_v(h_V_subgraph)
        h_E_subgraph = self.W_e(h_E_subgraph)
        for layer in self.layers:
            h_V_subgraph = layer(h_V_subgraph, edge_index_subgraph, h_E_subgraph)
        out_subgraph = self.W_out(h_V_subgraph)
        out_subgraph = scatter_mean(out_subgraph, batch_subgraph, dim=0)
        
        labels = labels.float()
        out = torch.cat([out_parent, out_subgraph], dim=1)
        logits = self.dense(out)
        # logits = self.attention_classifier(out_parent, out_subgraph)
        loss = self.loss_fn(logits.squeeze(-1), labels)
        return loss, logits
    
    def get_embedding(self, h_V, edge_index, h_E, batch):
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        for layer in self.layers:
            h_V = layer(h_V, edge_index, h_E)
        out = self.W_out(h_V)
        out = scatter_mean(out, batch, dim=0)
        return out