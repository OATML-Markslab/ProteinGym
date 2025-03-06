import torch
from torch import nn

from torchdrug import core, tasks
from torchdrug.core import Registry as R

from s3f import gvp
    

@R.register("tasks.ResidueTypePrediction")
class ResidueTypePrediction(tasks.AttributeMasking, core.Configurable):

    def __init__(self, model, mask_rate=0.15, dropout=0.5, graph_construction_model=None, plddt_threshold=None):
        super(ResidueTypePrediction, self).__init__(model, mask_rate=mask_rate, num_mlp_layer=1, graph_construction_model=graph_construction_model)
        if hasattr(self.model, "node_output_dim"):
            model_output_dim = self.model.node_output_dim
        else:
            model_output_dim = self.model.output_dim
        num_label = 20
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(model_output_dim, num_label)
        self.plddt_threshold = plddt_threshold

    def preprocess(self, train_set, valid_set, test_set):
        return

    def predict_and_target(self, batch, all_loss=None, metric=None):
        graph = batch["graph"]
        if self.graph_construction_model:
            graph = self.graph_construction_model.apply_node_layer(graph)

        # Random select residues to be masked
        num_nodes = graph.num_residues
        num_cum_nodes = num_nodes.cumsum(0)
        num_samples = (num_nodes * self.mask_rate).long().clamp(1)
        num_sample = num_samples.sum()
        sample2graph = torch.repeat_interleave(num_samples)
        node_index = (torch.rand(num_sample, device=self.device) * num_nodes[sample2graph]).long()
        node_index = node_index + (num_cum_nodes - num_nodes)[sample2graph]
        node_index = node_index.clamp(max=num_cum_nodes[-1]-1)

        target = graph.residue_type[node_index]
        mask_id = self.model.sequence_model.alphabet.get_idx("<mask>")
        with graph.residue():
            graph.residue_feature[node_index] = 0
            graph.residue_type[node_index] = mask_id

        # 80% mask, 10% replace, 10% unchange
        replace_prob = torch.rand_like(target.float())
        replace_mask = replace_prob < 0.1
        mutant_residue_type = torch.randint_like(replace_mask.long(), 20)
        with graph.residue():
            graph.residue_feature[node_index[replace_mask], mutant_residue_type[replace_mask]] = 1
            graph.residue_type[node_index[replace_mask]] = mutant_residue_type[replace_mask]
        unchanged_mask = (replace_prob < 0.2) & (replace_prob >= 0.1)
        with graph.residue():
            graph.residue_feature[node_index[unchanged_mask], target[unchanged_mask]] = 1
            graph.residue_type[node_index[unchanged_mask]] = target[unchanged_mask]

        if self.graph_construction_model:
           graph = self.graph_construction_model.apply_edge_layer(graph)
        input = graph.residue_feature.float()
        if isinstance(self.model.structure_model, gvp.SurfGVP):
            output = self.model(graph, input, batch["surf_graph"], all_loss, metric)
        else:
            output = self.model(graph, input, all_loss, metric)
        node_feature = output["node_feature"][node_index]

        node_feature = self.dropout(node_feature)
        pred = self.linear(node_feature)

        return pred, target

    def inference(self, batch):
        graph = batch["graph"]

        if self.graph_construction_model:
           graph = self.graph_construction_model(graph)
        input = graph.residue_feature.float()

        if isinstance(self.model.structure_model, gvp.SurfGVP):
            output = self.model(graph, input, batch["surf_graph"])
        else:
            output = self.model(graph, input)
        node_feature = output["node_feature"]
        pred = self.linear(node_feature)

        if self.plddt_threshold:
            # for AlphaFold2-predicted structure pdb files, plddt is saved as b_factor
            plddt_mask = batch["graph"].b_factor < self.plddt_threshold
            pred[plddt_mask] = output["sequence_logits"][plddt_mask]

        return pred, graph.num_residues