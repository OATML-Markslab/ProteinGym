
import os
import torch
import sys
import math
import random
import warnings
import torch
import os
import sys
import torch.nn.functional as F
import scipy.spatial as spa
import numpy as np
from tqdm import tqdm
from scipy.special import softmax
from Bio.PDB import PDBParser, ShrakeRupley
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from rdkit.Chem import GetPeriodicTable
from typing import Callable, List, Optional
from torch.utils.data import DataLoader
from torch_geometric.data import InMemoryDataset, Data
current_dir = os.getcwd()
sys.path.append(current_dir)
from src.utils.dataset_utils import safe_index, one_hot_res, log, dihedral, NormalizeProtein, dataset_argument_, get_stat

cwd = os.getcwd()

sys.path.append(cwd + '/src/dataset_utils')
warnings.filterwarnings("ignore")

one_letter = {
    'VAL':'V', 'ILE':'I', 'LEU':'L', 'GLU':'E', 'GLN':'Q',
    'ASP':'D', 'ASN':'N', 'HIS':'H', 'TRP':'W', 'PHE':'F', 'TYR':'Y',
    'ARG':'R', 'LYS':'K', 'SER':'S', 'THR':'T', 'MET':'M', 'ALA':'A',
    'GLY':'G', 'PRO':'P', 'CYS':'C'
    }

class CathDataset(InMemoryDataset):
    r"""
    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset.
        raw_dir (string, optional): Root directory where the
        original dataset stored(default: :obj:`None`)

        num_residue_type (int, optional): The number of amino acid types.
        (default: obj:'20')
        micro_radius (int, optional): The radius of micro-environment
        centered on the mask node. (default: obj:'20')
        c_alpha_max_neighbors (int, optional): The number of maximum
        connected nodes. (default: obj:'10')
        cutoff (int, optional): The maximum connected nodes distance
        (default: obj:'30')
        seq_dist_cut (int, optional): one-hot encoding the sequence distance
        edge attribute
        (default: obj:)
        [0.25,0.5,0.75,0.9,0.95,0.98,0.99]
        [  2.   3.  13.  63. 127. 247. 347.]
        num_val (int, optional): The number of validation samples in case of "random" split. (default: 500)
        num_test (int, optional): The number of test samples in case of "random" split. (default: 1000)

        # use_localdatastet (bool) (bool,optional): If :obj:'True', online dataset
        # will be downloaded. If not, local pdb files will be used
        # (default: obj:'True')

        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    splits = ['train', 'val', 'test']
    allowable_features = {
        'possible_atomic_num_list': list(range(1, 119)) + ['misc'],
        'possible_chirality_list': [
            'CHI_UNSPECIFIED',
            'CHI_TETRAHEDRAL_CW',
            'CHI_TETRAHEDRAL_CCW',
            'CHI_OTHER'
        ],
        'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
        'possible_numring_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
        'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
        'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
        'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
        'possible_hybridization_list': [
            'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
        ],
        'possible_is_aromatic_list': [False, True],
        'possible_is_in_ring3_list': [False, True],
        'possible_is_in_ring4_list': [False, True],
        'possible_is_in_ring5_list': [False, True],
        'possible_is_in_ring6_list': [False, True],
        'possible_is_in_ring7_list': [False, True],
        'possible_is_in_ring8_list': [False, True],
        'possible_amino_acids': ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS',
                                 'MET',
                                 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'HIP', 'HIE', 'TPO', 'HID', 'LEV',
                                 'MEU',
                                 'PTR', 'GLV', 'CYT', 'SEP', 'HIZ', 'CYM', 'GLM', 'ASQ', 'TYS', 'CYX', 'GLZ', 'misc'],
        'possible_atom_type_2': ['C*', 'CA', 'CB', 'CD', 'CE', 'CG', 'CH', 'CZ', 'N*', 'ND', 'NE', 'NH', 'NZ', 'O*',
                                 'OD',
                                 'OE', 'OG', 'OH', 'OX', 'S*', 'SD', 'SG', 'misc'],
        'possible_atom_type_3': ['C', 'CA', 'CB', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'CE3', 'CG', 'CG1', 'CG2',
                                 'CH2',
                                 'CZ', 'CZ2', 'CZ3', 'N', 'ND1', 'ND2', 'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'NZ', 'O',
                                 'OD1',
                                 'OD2', 'OE1', 'OE2', 'OG', 'OG1', 'OH', 'OXT', 'SD', 'SG', 'misc'],
    }

    def __init__(self, root: str,
                 split: str = 'train',
                 num_residue_type: int = 20,
                 micro_radius: int = 20,
                 c_alpha_max_neighbors: int = 10,
                 cutoff: int = 30,
                 seq_dist_cut: int = 64,
                 use_micro: bool = False,
                 use_angle: bool = False,
                 use_omega: bool = False,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 divide_num: int = 1,
                 divide_idx: int = 0,
                 set_length: int = 500,
                 num_val: int = 10,
                 is_normalize: bool = True,
                 normalize_file: str = None,
                 p: float = 0.5,
                 use_sasa: bool =False,
                 use_bfactor: bool = False,
                 use_dihedral: bool = False,
                 use_coordinate: bool = False,
                 use_denoise: bool = False,
                 noise_type: str = 'wild',
                 temperature = 1.0
                 ):
        self.p=p
        self.use_sasa=use_sasa
        self.use_bfactor=use_bfactor
        self.use_dihedral=use_dihedral
        self.use_coordinate=use_coordinate
        self.use_denoise=use_denoise
        self.noise_type = noise_type
        self.temperature = temperature
        
        self.split = split
        assert self.split in self.splits

        self.num_residue_type = num_residue_type
        self.micro_radius = micro_radius
        self.c_alpha_max_neighbors = c_alpha_max_neighbors
        self.seq_dist_cut = seq_dist_cut
        self.use_micro = use_micro
        self.use_angle = use_angle
        self.use_omega = use_omega
        self.cutoff = cutoff

        self.num_val = num_val
        self.divide_num = divide_num
        self.divide_idx = divide_idx
        self.set_length = set_length

        self.is_normalize = is_normalize
        self.normalize_file = normalize_file

        self.wrong_proteins = ['1kp0A01', '2atcA02']

        self.sr = ShrakeRupley(probe_radius=1.4,  # in A. Default is 1.40 roughly the radius of a water molecule.
                               n_points=100)  # resolution of the surface of each atom. Default is 100. A higher number of points results in more precise measurements, but slows down the calculation.
        self.periodic_table = GetPeriodicTable()
        self.biopython_parser = PDBParser()

        super().__init__(root, transform, pre_transform, pre_filter)
        self.dataset = torch.load(self.processed_paths[self.splits.index(self.split)])
        # self.data, self.slices = torch.load(
        #     self.processed_paths[self.splits.index(self.split)])
        # self.nums_amino_cum = self.slices['x']

    @property
    def raw_file_names(self) -> str:
        raw_file_names = os.path.join('data', 'cath', "dompdb")
        if not os.path.exists(raw_file_names):
            os.mkdir(raw_file_names)
        return raw_file_names

    @property
    def raw_dir(self) -> str:
        if not os.path.exists(self.root):
            os.mkdir(self.root)
        raw_dir = os.path.join(self.root, 'raw')
        if not os.path.exists(raw_dir):
            os.mkdir(raw_dir)
        return raw_dir

    @property
    def saved_graph_dir(self) -> str:
        dir_root = os.path.join(self.root)
        if not os.path.exists(dir_root):
            os.mkdir(dir_root)
        dir_name = os.path.join(dir_root, 'graph_seq')
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        if not self.set_length:
            self.set_length = len(os.listdir(dir_name))
        return dir_name

    @property
    def saved_amino_cum(self) -> str:
        amino_cum_name = os.path.join(
            self.root, 'amino_cum.pt')
        return amino_cum_name

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'processed_seq')

    @property
    def processed_file_names(self) -> str:
        return ['train.pt', 'val.pt']


    def write_info(self):
        written_filename = os.path.join(self.root, 'wrong_protein_names.txt')
        file = open(written_filename, "w+")
        for protein_name in self.wrong_proteins:
            file.writelines(protein_name + '\n')
        file.close()

    def process(self):
        #generate graph data and save in graph dir
        self.generate_protein_graph()
        # self.write_info()

        filenames = os.listdir(self.saved_graph_dir)
        protein_length = len(filenames)
        if self.set_length:
            protein_length = min(protein_length, self.set_length)

        if not self.normalize_file:
            self.normalize_file = get_stat(self.saved_graph_dir)

        random.shuffle(filenames)
        train_list = [f for f in filenames if "_" in f or "-" in f]
        filenames = [f for f in filenames if "_" not in f or "-" not in f]
        train_list.extend(filenames[:-self.num_val])
        filenames_list = [train_list, filenames[-self.num_val:]]
        
        for k in range(2):####split train,val,test
            data_list = []

            ###move special name to test set
            special_name_list = ["p53-dimer.pdb.pt"]
            for special_name in special_name_list:
                if special_name in filenames_list[0]:
                    filenames_list[0].remove(special_name)
                    filenames_list[1].append(special_name)
            for i in tqdm(range(len(filenames_list[k]))):
                file = filenames_list[k][i]
                try:
                    graph1 = torch.load(os.path.join(self.saved_graph_dir, file))##load processed graph data torch pt file
                except:
                    print(file)
                    continue
                del graph1['distances']
                del graph1['edge_dist']
                del graph1['mu_r_norm']
                del graph1['seq']
                data_list.append(graph1)
            if self.is_normalize:
                normalize_transform = NormalizeProtein(filename=self.normalize_file)
                data_list = [d for d in data_list if normalize_transform(d)]
            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]
            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            torch.save(data_list, self.processed_paths[k])

    def generate_protein_graph(self):
        names = os.listdir(self.raw_file_names)
        print(names)
        names.sort()
        n = int(np.ceil(len(names) / self.divide_num))
        names = names[n * self.divide_idx:min(len(names), n * (self.divide_idx + 1))]
        for idx, name in enumerate(tqdm(names)):
            saved_graph_filename = os.path.join(self.saved_graph_dir, name + '.pt')
            if os.path.exists(saved_graph_filename):
                continue
            protein_filename = os.path.join(self.raw_file_names, name)
            if (name in self.wrong_proteins) or (not protein_filename):
                continue
            try:
                rec, rec_coords, c_alpha_coords, n_coords, c_coords,seq = self.get_receptor_inference(protein_filename)
            except:
                continue
            if rec !=False:
                    if len(seq)>len(c_alpha_coords):
                        del seq[-(len(seq)-len(c_alpha_coords)):]
                    #meet "dna" data will remove the file and rec will be false
            # print(self.c_alpha_max_neighbors)
            rec_graph = self.get_calpha_graph(rec, c_alpha_coords, n_coords, c_coords, rec_coords,seq)
            if not rec_graph:
                self.wrong_proteins.append(name)
                continue
            torch.save(rec_graph, saved_graph_filename)

    def rec_residue_featurizer(self, rec, chain_id, one_hot=True, add_feature=None):
        count = 0
        flag_sasa=1
        try:
            self.sr.compute(rec, level="R")
        except:
            flag_sasa=0
        for i, chain in enumerate(rec.get_chains()):
            if i != chain_id:
                continue
            num_res = len(list(chain.get_residues()))#len([_ for _ in rec.get_residues()])
            num_feature = 2
            if add_feature.any():
                num_feature += add_feature.shape[1]
            res_feature = torch.zeros(num_res, self.num_residue_type + num_feature)
            for i, residue in enumerate(chain.get_residues()):
                if flag_sasa==0:
                    residue.sasa=0
                sasa = residue.sasa
                for atom in residue:
                    if atom.name == 'CA':
                        bfactor = atom.bfactor
                assert not np.isinf(bfactor)
                assert not np.isnan(bfactor)
                assert not np.isinf(sasa)
                assert not np.isnan(sasa)

                residx = safe_index(
                    self.allowable_features['possible_amino_acids'], residue.get_resname())
                res_feat_1 = one_hot_res(
                    residx, num_residue_type=self.num_residue_type) if one_hot else [residx]
                if not res_feat_1:
                    return False
                res_feat_1.append(sasa)
                res_feat_1.append(bfactor)
                if num_feature > 2:
                    res_feat_1.extend(list(add_feature[count, :]))
                res_feature[count, :] = torch.tensor(res_feat_1, dtype=torch.float32)
                count += 1
        # print("numnodes:", num_res, count,len(list(chain.get_residues())))
        for k in range(self.num_residue_type, self.num_residue_type + 2):
            mean = res_feature[:, k].mean()
            std = res_feature[:, k].std()
            res_feature[:, k] = (res_feature[:, k] -mean) / (std + 0.000000001)
        return res_feature

    def get_node_features(self, n_coords, c_coords, c_alpha_coords, coord_mask, with_coord_mask=True, use_angle=False,
                          use_omega=False):
        num_res = n_coords.shape[0]
        if use_omega:
            num_angle_type = 3
            angles = np.zeros((num_res, num_angle_type))
            for i in range(num_res - 1):
                # These angles are called φ (phi) which involves the backbone atoms C-N-Cα-C
                angles[i, 0] = dihedral(
                    c_coords[i], n_coords[i], c_alpha_coords[i], n_coords[i + 1])
                # psi involves the backbone atoms N-Cα-C-N.
                angles[i, 1] = dihedral(
                    n_coords[i], c_alpha_coords[i], c_coords[i], n_coords[i + 1])
                angles[i, 2] = dihedral(
                    c_alpha_coords[i], c_coords[i], n_coords[i + 1], c_alpha_coords[i + 1])
        else:
            num_angle_type = 2
            angles = np.zeros((num_res, num_angle_type))
            for i in range(num_res - 1):
                # These angles are called φ (phi) which involves the backbone atoms C-N-Cα-C
                angles[i, 0] = dihedral(
                    c_coords[i], n_coords[i], c_alpha_coords[i], n_coords[i + 1])
                # psi involves the backbone atoms N-Cα-C-N.
                angles[i, 1] = dihedral(
                    n_coords[i], c_alpha_coords[i], c_coords[i], n_coords[i + 1])
        if use_angle:
            node_scalar_features = angles
        else:
            node_scalar_features = np.zeros((num_res, num_angle_type * 2))
            for i in range(num_angle_type):
                node_scalar_features[:, 2 * i] = np.sin(angles[:, i])
                node_scalar_features[:, 2 * i + 1] = np.cos(angles[:, i])

        if with_coord_mask:
            node_scalar_features = torch.cat([
                node_scalar_features,
                coord_mask.float().unsqueeze(-1)
            ], dim=-1)
        node_vector_features = None
        return node_scalar_features, node_vector_features

    def get_calpha_graph(self, rec, c_alpha_coords, n_coords, c_coords, coords, seq):
        chain_id = 0
        scalar_feature, vec_feature = self.get_node_features(n_coords, c_coords, c_alpha_coords, coord_mask=None, with_coord_mask=False, use_angle=self.use_angle, use_omega=self.use_omega)
        # Extract 3D coordinates and n_i,u_i,v_i
        # vectors of representative residues ################
        residue_representatives_loc_list = []
        n_i_list = []
        u_i_list = []
        v_i_list = []
        for i, chain in enumerate(rec.get_chains()):
            if i != chain_id:
                continue
            for i, residue in enumerate(chain.get_residues()):
                n_coord = n_coords[i]
                c_alpha_coord = c_alpha_coords[i]
                c_coord = c_coords[i]
                u_i = (n_coord - c_alpha_coord) / \
                    np.linalg.norm(n_coord - c_alpha_coord)
                t_i = (c_coord - c_alpha_coord) / \
                    np.linalg.norm(c_coord - c_alpha_coord)
                n_i = np.cross(u_i, t_i) / \
                    np.linalg.norm(np.cross(u_i, t_i))   # main chain
                v_i = np.cross(n_i, u_i)
                assert (math.fabs(
                    np.linalg.norm(v_i) - 1.) < 1e-5), "protein utils protein_to_graph_dips, v_i norm larger than 1"
                n_i_list.append(n_i)
                u_i_list.append(u_i)
                v_i_list.append(v_i)
                residue_representatives_loc_list.append(c_alpha_coord)

        residue_representatives_loc_feat = np.stack(residue_representatives_loc_list, axis=0)  # (N_res, 3)
        n_i_feat = np.stack(n_i_list, axis=0)
        u_i_feat = np.stack(u_i_list, axis=0)
        v_i_feat = np.stack(v_i_list, axis=0)
        num_residues = len(c_alpha_coords)
        if num_residues <= 1:
            raise ValueError(f"rec contains only 1 residue!")
        ################### Build the k-NN graph ##############################
        assert num_residues == residue_representatives_loc_feat.shape[0]
        assert residue_representatives_loc_feat.shape[1] == 3
        distances = spa.distance.cdist(c_alpha_coords, c_alpha_coords)

        src_list = []
        dst_list = []
        dist_list = []
        mean_norm_list = []
        for i in range(num_residues):
            dst = list(np.where(distances[i, :] < self.cutoff)[0])
            dst.remove(i)
            if self.c_alpha_max_neighbors != None and len(dst) > self.c_alpha_max_neighbors:
                dst = list(np.argsort(distances[i, :]))[
                    1: self.c_alpha_max_neighbors + 1]
            if len(dst) == 0:
                # choose second because first is i itself
                dst = list(np.argsort(distances[i, :]))[1:2]
                log(
                    f'The c_alpha_cutoff {self.cutoff} was too small for one c_alpha such that it had no neighbors. So we connected it to the closest other c_alpha')
            assert i not in dst
            src = [i] * len(dst)
            src_list.extend(src)
            dst_list.extend(dst)
            valid_dist = list(distances[i, dst])
            dist_list.extend(valid_dist)
            valid_dist_np = distances[i, dst]
            sigma = np.array([1., 2., 5., 10., 30.]).reshape((-1, 1))
            weights = softmax(- valid_dist_np.reshape((1, -1))** 2 / sigma, axis=1)  # (sigma_num, neigh_num)
            # print(weights) why weight??
            assert weights[0].sum() > 1 - 1e-2 and weights[0].sum() < 1.01
            diff_vecs = residue_representatives_loc_feat[src, :] - residue_representatives_loc_feat[dst, :]  # (neigh_num, 3)
            mean_vec = weights.dot(diff_vecs)  # (sigma_num, 3)
            denominator = weights.dot(np.linalg.norm(diff_vecs, axis=1))  # (sigma_num,)
            mean_vec_ratio_norm = np.linalg.norm(mean_vec, axis=1) / denominator  # (sigma_num,)
            mean_norm_list.append(mean_vec_ratio_norm)
        assert len(src_list) == len(dst_list)
        assert len(dist_list) == len(dst_list)
        residue_representatives_loc_feat = torch.from_numpy(residue_representatives_loc_feat.astype(np.float32))
        x = self.rec_residue_featurizer(rec, chain_id, one_hot=True, add_feature=scalar_feature)
        if isinstance(x, bool) and (not x):
            return False
        ######key part to generate graph!!!!!main
        graph = Data(
            x=x,## 26 feature 20+sasa+b factor+ two face angle
            pos=residue_representatives_loc_feat,
            edge_attr=self.get_edge_features(src_list, dst_list, dist_list, divisor=4), ##edge features
            edge_index=torch.tensor([src_list, dst_list]),
            edge_dist=torch.tensor(dist_list),
            distances=torch.tensor(distances),
            mu_r_norm=torch.from_numpy(np.array(mean_norm_list).astype(np.float32)),
            seq = seq) ##about density capture
        # Loop over all edges of the graph and build the various p_ij, q_ij, k_ij, t_ij pairs
        edge_feat_ori_list = []
        for i in range(len(dist_list)):
            src = src_list[i]
            dst = dst_list[i]
            # place n_i, u_i, v_i as lines in a 3x3 basis matrix
            basis_matrix = np.stack(
                (n_i_feat[dst, :], u_i_feat[dst, :], v_i_feat[dst, :]), axis=0)
            p_ij = np.matmul(basis_matrix,residue_representatives_loc_feat[src, :] - residue_representatives_loc_feat[dst, :])
            q_ij = np.matmul(basis_matrix, n_i_feat[src, :])  # shape (3,)
            k_ij = np.matmul(basis_matrix, u_i_feat[src, :])
            t_ij = np.matmul(basis_matrix, v_i_feat[src, :])
            s_ij = np.concatenate((p_ij, q_ij, k_ij, t_ij), axis=0)  # shape (12,)
            edge_feat_ori_list.append(s_ij)

        edge_feat_ori_feat = np.stack(edge_feat_ori_list, axis=0)  # shape (num_edges, 4, 3)
        edge_feat_ori_feat = torch.from_numpy(edge_feat_ori_feat.astype(np.float32))
        graph.edge_attr = torch.cat([graph.edge_attr, edge_feat_ori_feat], axis=1)  # (num_edges, 17)
        # graph = self.remove_node(graph, graph.x.shape[0]-1)###remove the last node, can not calculate the two face angle
        # self.get_calpha_graph_single(graph, 6)
        return graph

    def remove_node(self, graph, node_idx):
        new_graph = Data.clone(graph)
        # delete node
        new_graph.x = torch.cat(
            [new_graph.x[:node_idx, :], new_graph.x[node_idx + 1:, :]])
        new_graph.pos = torch.cat(
            [new_graph.pos[:node_idx, :], new_graph.pos[node_idx + 1:, :]])
        new_graph.mu_r_norm = torch.cat(
            [new_graph.mu_r_norm[:node_idx, :], new_graph.mu_r_norm[node_idx + 1:, :]])

        # delete edge
        keep_edge = (torch.sum(new_graph.edge_index == node_idx, dim=0) == 0)
        new_graph.edge_index = new_graph.edge_index[:, keep_edge]
        new_graph.edge_attr = new_graph.edge_attr[keep_edge, :]
        return new_graph

    def get_edge_features(self, src_list, dst_list, dist_list, divisor=4):
        seq_edge = torch.absolute(torch.tensor(
            src_list) - torch.tensor(dst_list)).reshape(-1, 1)
        seq_edge = torch.where(seq_edge > self.seq_dist_cut,
                               self.seq_dist_cut, seq_edge)
        seq_edge = F.one_hot(
            seq_edge, num_classes=self.seq_dist_cut + 1).reshape((-1, self.seq_dist_cut + 1))
        contact_sig = torch.where(torch.tensor(
            dist_list) <= 8, 1, 0).reshape(-1, 1)
        # avg distance = 7. So divisor = (4/7)*7 = 4
        dist_fea = self.distance_featurizer(dist_list, divisor=divisor)
        return torch.concat([seq_edge, dist_fea, contact_sig], dim=-1)

    def get_receptor_inference(self, rec_path):
        chain_id=0
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=PDBConstructionWarning)
            structure = self.biopython_parser.get_structure('random_id', rec_path)
            rec = structure[0]##len(structure)=1
            head = self.biopython_parser.get_header()['head']
            if head.find('dna') > -1:
                return False, False, False, False, False,False
        coords = []
        c_alpha_coords = []
        n_coords = []
        c_coords = []
        valid_chain_ids = []
        lengths = []
        seq = []
        for i, chain in enumerate(rec):
            print("chain num",i,chain_id,chain)
            if i != chain_id:##select chain A:i=0 or B:i=1
                continue
            chain_coords = []  # num_residues, num_atoms, 3
            chain_c_alpha_coords = []
            chain_n_coords = []
            chain_c_coords = []
            count = 0
            invalid_res_ids = []
            for res_idx, residue in enumerate(chain):
                if residue.get_resname() == 'HOH':
                    invalid_res_ids.append(residue.get_id())
                    continue
                residue_coords = []
                c_alpha, n, c = None, None, None
                for atom in residue:
                    if atom.name == 'CA':
                        c_alpha = list(atom.get_vector())
                        seq.append(str(residue).split(" ")[1])
                    if atom.name == 'N':
                        n = list(atom.get_vector())
                    if atom.name == 'C':
                        c = list(atom.get_vector())
                    residue_coords.append(list(atom.get_vector()))
                # only append residue if it is an amino acid and not some weired molecule that is part of the complex
                if c_alpha != None and n != None and c != None:
                    chain_c_alpha_coords.append(c_alpha)
                    chain_n_coords.append(n)
                    chain_c_coords.append(c)
                    chain_coords.append(np.array(residue_coords))
                    count += 1
                else:
                    invalid_res_ids.append(residue.get_id())
            for res_id in invalid_res_ids:
                chain.detach_child(res_id)
            lengths.append(count)
            coords.append(chain_coords)
            c_alpha_coords.append(np.array(chain_c_alpha_coords))
            n_coords.append(np.array(chain_n_coords))
            c_coords.append(np.array(chain_c_coords))
            if len(chain_coords) > 0:
                valid_chain_ids.append(chain.get_id())
        valid_coords = []
        valid_c_alpha_coords = []
        valid_n_coords = []
        valid_c_coords = []
        valid_lengths = []
        invalid_chain_ids = []
        for i, chain in enumerate(rec):
            # print("chain:",i,chain, len(valid_coords), len(valid_chain_ids), len(coords), coords[0][0].shape, len(coords[0]))
            if i != chain_id:
                continue
            if chain.get_id() in valid_chain_ids:
                valid_coords.append(coords[0])
                valid_c_alpha_coords.append(c_alpha_coords[0])
                valid_n_coords.append(n_coords[0])
                valid_c_coords.append(c_coords[0])
                valid_lengths.append(lengths[0])
            else:
                invalid_chain_ids.append(chain.get_id())
        # list with n_residues arrays: [n_atoms, 3]
        coords = [item for sublist in valid_coords for item in sublist]
        if len(valid_c_alpha_coords) == 0:
            return False, False, False, False, False,False
        c_alpha_coords = np.concatenate(valid_c_alpha_coords, axis=0)  # [n_residues, 3]
        n_coords = np.concatenate(valid_n_coords, axis=0)  # [n_residues, 3]
        c_coords = np.concatenate(valid_c_coords, axis=0)  # [n_residues, 3]

        for invalid_id in invalid_chain_ids:
            rec.detach_child(invalid_id)

        assert len(c_alpha_coords) == len(n_coords)
        assert len(c_alpha_coords) == len(c_coords)
        assert sum(valid_lengths) == len(c_alpha_coords)
        return rec, coords, c_alpha_coords, n_coords, c_coords,seq

    def len(self):
        return len(self.dataset)

    def get_statistic_info(self):
        node_num = torch.zeros(self.length_total)
        edge_num = torch.zeros(self.length_total)
        for i in tqdm(range(self.length_total)):
            graph = self.get(i)
            node_num[i] = graph.x.shape[0]
            edge_num[i] = graph.edge_index.shape[1]
        num_node_min = torch.min(node_num)
        num_node_max = torch.max(node_num)
        num_node_avg = torch.mean(node_num)
        num_edge_min = torch.min(edge_num)
        num_edge_max = torch.max(edge_num)
        num_edge_avg = torch.mean(edge_num)
        print(f'Graph Num: {self.length_total}')
        print(
            f'Min Nodes: {num_node_min:.2f} Max Nodes: {num_node_max:.2f}. Avg Nodes: {num_node_avg:.2f}')
        print(
            f'Min Edges: {num_edge_min:.2f} Max Edges: {num_edge_max:.2f}. Avg Edges: {num_edge_avg:.2f}')

    def _get_noise(self, token_len: int, prob: List=[]):
        prob = prob if prob else [0.08, 0.05, 0.04, 0.06, 0.01, 0.04, 0.07, 0.07, 0.02, 0.06, 0.1, 0.06,
                                  0.02, 0.04, 0.04, 0.06, 0.05, 0.01, 0.03, 0.07]
        multant_pos = ((torch.rand(token_len) <= self.p)).nonzero().flatten()
        if len(multant_pos) == 0:
            return None, None
        multant_trg = torch.multinomial(torch.tensor(prob), len(multant_pos), replacement=True)
        return multant_pos, multant_trg
        
    
    def _token_rep_noise(self, data, multant_pos, multant_trg, rep_noise_type='window_3'):
        num_classes = 20
        multant_rep = data.token_rep.clone()
        for mut_pos, mut_trg in zip(multant_pos, multant_trg):
            mut_trg_ = F.one_hot(mut_trg, num_classes=num_classes)
            if rep_noise_type == 'mean':
                trg_rep = data.token_rep[(data.x[:,:20] == mut_trg_).sum(1) == num_classes].mean(0)
                if torch.isnan(trg_rep).sum() > 0:
                    continue
                multant_rep[mut_pos] = trg_rep
            elif "window" in rep_noise_type:
                window_size = int(rep_noise_type.split("_")[-1])
                start_pos = mut_pos - math.ceil(window_size/2)
                end_pos = start_pos + window_size
                if end_pos > len(data.token_rep):
                    start_pos = mut_pos - window_size
                    trg_rep = data.token_rep[start_pos:].mean(0)
                elif start_pos < 0:
                    end_pos = window_size
                    trg_rep = data.token_rep[:end_pos].mean(0)
                else:
                    trg_rep = data.token_rep[start_pos:end_pos].mean(0)
                multant_rep[mut_pos] = trg_rep
        return multant_rep

    def get(self, idx):
        # idx_protein = idx
        # idx_x0, idx_x1 = self.slices['x'][idx_protein], self.slices['x'][idx_protein + 1]
        # idx_edge0, idx_edge1 = self.slices['edge_index'][idx_protein], self.slices['edge_index'][idx_protein + 1]
        
        # data = Data(
        #     x=self.data.x[idx_x0:idx_x1, :],
        #     pos=self.data.pos[idx_x0:idx_x1, :],
        #     edge_index=self.data.edge_index[:, idx_edge0:idx_edge1],
        #     edge_attr=self.data.edge_attr[idx_edge0:idx_edge1, :],
        #     lenth=idx_x1-idx_x0
        # )
        data = self.dataset[idx]

        token_len = data.x.shape[0]
        data.y = data.x[:token_len, :self.num_residue_type].argmax(1)
        multant_pos, multant_trg = self._get_noise(token_len=token_len)
        if multant_pos is not None:
            noisey = data.x[:, :20].argmax(dim=1)
            noisey[multant_pos] = multant_trg
            data.x[:,:20] = F.one_hot(noisey, num_classes=20)
        
        return data
    

    def find_idx(self, idx_protein, amino_idx):
        idx = (self.distances[idx_protein][:-1, amino_idx]< self.micro_radius).nonzero(as_tuple=True)[0]
        return idx
    
    def get_calpha_graph_single(self, graph, idx_protein, amino_idx):
        choosen_amino_idx = self.find_idx(idx_protein, amino_idx)
        keep_edge_index = []
        for edge_idx in range(graph.num_edges):
            edge = graph.edge_index.t()[edge_idx]
            if (edge[0] in choosen_amino_idx) and (edge[1] in choosen_amino_idx):
                keep_edge_index.append(edge_idx)
        graph1 = Data(x=graph.x[choosen_amino_idx, :],
                      pos=graph.pos[choosen_amino_idx, :],
                      edge_index=graph.edge_index[:, keep_edge_index],
                      edge_attr=graph.edge_attr[keep_edge_index, :],
                      mu_r_norm=graph.mu_r_norm[choosen_amino_idx, :])
        return graph1
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
    
    def distance_featurizer(self, dist_list, divisor) -> torch.Tensor:
        # you want to use a divisor that is close to 4/7 times the average distance that you want to encode
        length_scale_list = [1.5 ** x for x in range(15)]
        center_list = [0. for _ in range(15)]
        num_edge = len(dist_list)
        dist_list = np.array(dist_list)
        transformed_dist = [np.exp(- ((dist_list / divisor) ** 2) / float(length_scale))
                            for length_scale, center in zip(length_scale_list, center_list)]
        transformed_dist = np.array(transformed_dist).T
        transformed_dist = transformed_dist.reshape((num_edge, -1))
        return torch.from_numpy(transformed_dist.astype(np.float32))
        
    