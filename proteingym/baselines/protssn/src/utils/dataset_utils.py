import torch
import sys
import os
import random
import csv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from numpy import array, cross, pi, arccos, sqrt
from tqdm import tqdm
from time import time
from torch_geometric.transforms import BaseTransform
from scipy.stats import spearmanr
from numpy import nan

def dataset_argument_(root):
    dataset_arg = {}
    if root == "cath40_k10_dyn_imem":
        dataset_arg['root'] = f"data/{root}"
        dataset_arg['name'] = '40'
        dataset_arg['set_length'] = None
        dataset_arg['normal_file'] = None
        dataset_arg['divide_num'] = 1
        dataset_arg['divide_idx'] = 0
        dataset_arg['c_alpha_max_neighbors'] = 10
    return dataset_arg

def get_stat(graph_root, limited_num=None, num_subgroup=1000, max_limits=100000):
    # obtain mean and std of graphs in graph_root
    # graph_root: string, calculate mean and std of all attributes of graphs in graph_root
    # limited_num: int, optional, just calculated limited number of graphs in graph_root
    # num_Subgroup: int, group all graphs in graph_root, the number of each subgroup is num_subgroup
    # max_limits: int, set the initial minimum value as max_limits

    wrong_proteins = []
    filenames = os.listdir(graph_root)
    random.shuffle(filenames)
    # set sample length
    n = len(filenames)
    if limited_num:
        n = min(n, limited_num)
    count = 0
    if n < num_subgroup * 10:
        num_subgroup = 1

    # initialize scalar value
    num_node_min, num_edge_min = torch.tensor(
        [max_limits]), torch.tensor([max_limits])
    num_node_max, num_node_avg, num_edge_max, num_edge_avg = torch.tensor(
        [0]), torch.tensor([0]), torch.tensor([0]), torch.tensor([0])

    # initialize mean, std
    graph = torch.load(os.path.join(graph_root, filenames[0]))
    x, pos, edge_attr = graph.x, graph.pos, graph.edge_attr
    x_mean = torch.zeros(x.shape[1])
    x_max = torch.zeros(x.shape[1])
    x_min = torch.tensor([max_limits for i in range(x.shape[1])])
    x_std = torch.zeros(x.shape[1])
    pos_mean = torch.zeros(pos.shape[1])
    pos_std = torch.zeros(pos.shape[1])
    edge_attr_mean = torch.zeros(edge_attr.shape[1])
    edge_attr_std = torch.zeros(edge_attr.shape[1])

    # initialize sub mean, std
    x_mean_1 = torch.zeros(x.shape[1])
    x_std_1 = torch.zeros(x.shape[1])
    pos_mean_1 = torch.zeros(pos.shape[1])
    pos_std_1 = torch.zeros(pos.shape[1])

    edge_attr_mean_1 = torch.zeros(edge_attr.shape[1])
    edge_attr_std_1 = torch.zeros(edge_attr.shape[1])

    for i in tqdm(range(n)):
        file = filenames[i]
        graph = torch.load(os.path.join(graph_root, file))
        x, pos, mu_r_norm, edge_attr = graph.x, graph.pos, graph.mu_r_norm, graph.edge_attr
        if torch.isnan(x).any():
            wrong_proteins.append(file)
            continue
        count += 1
        node_num = graph.x.shape[0]
        edge_num = graph.edge_attr.shape[0]
        num_node_min = min(num_node_min, node_num)
        num_edge_min = min(num_edge_min, edge_num)
        num_node_max = max(num_node_max, node_num)
        num_edge_max = max(num_edge_max, edge_num)
        num_node_avg += node_num
        num_edge_avg += edge_num

        x_max = torch.max(x_max, x.max(axis=0).values)
        x_min = torch.min(x_min, x.min(axis=0).values)
        x_mean_1 += x.nanmean(axis=0)
        x_std_1 += x.std(axis=0)
        pos_mean_1 += pos.mean(axis=0)
        pos_std_1 += pos.std(axis=0)
        edge_attr_mean_1 += edge_attr.mean(axis=0)
        edge_attr_std_1 += edge_attr.std(axis=0)

        if count == num_subgroup:
            x_mean += x_mean_1.div_(num_subgroup)
            x_std += x_std_1.div_(num_subgroup)
            pos_mean += pos_mean_1.div_(num_subgroup)
            pos_std += pos_std_1.div_(num_subgroup)
            edge_attr_mean += edge_attr_mean_1.div_(num_subgroup)
            edge_attr_std += edge_attr_std_1.div_(num_subgroup)

            x_mean_1 = torch.zeros(x.shape[1])
            x_std_1 = torch.zeros(x.shape[1])
            pos_mean_1 = torch.zeros(pos.shape[1])
            pos_std_1 = torch.zeros(pos.shape[1])
            edge_attr_mean_1 = torch.zeros(edge_attr.shape[1])
            edge_attr_std_1 = torch.zeros(edge_attr.shape[1])
            count = 0

    num_node_avg = num_node_avg/n
    num_edge_avg = num_edge_avg/n
    n_2 = n // num_subgroup
    x_mean = x_mean.div_(n_2)
    x_std = x_std.div_(n_2)
    pos_mean = pos_mean.div_(n_2)
    pos_std = pos_std.div_(n_2)
    edge_attr_mean = edge_attr_mean.div_(n_2)
    edge_attr_std = edge_attr_std.div_(n_2)

    dic = {'x_max': x_max, 'x_min': x_min, 'x_mean': x_mean, 'x_std': x_std,
           'pos_mean': pos_mean, 'pos_std': pos_std,
           'edge_attr_mean': edge_attr_mean, 'edge_attr_std': edge_attr_std,
           'num_graph': n - len(wrong_proteins),
           'num_node_min': num_node_min, 'num_edge_min': num_edge_min,
           'num_node_max': num_node_max, 'num_edge_max': num_edge_max,
           'num_node_avg': num_node_avg, 'num_edge_avg': num_edge_avg}

    filename = 'mean_attr'
    saved_filename_pt = os.path.join(
        '/'.join(graph_root.split('/')[:-1]), filename + '.pt')
    torch.save(dic, saved_filename_pt)
    saved_filename = os.path.join(
        '/'.join(graph_root.split('/')[:-1]), filename + '.csv')
    w = csv.writer(open(saved_filename, 'w'))
    for key, val in dic.items():
        w.writerow([key, val])

    saved_filename = os.path.join(
        '/'.join(graph_root.split('/')[:-1]), filename + '_proteins.txt')
    with open(saved_filename, 'w') as f:
        for i in range(n):
            f.write(str(filenames[i]) + '\n')

    saved_filename = os.path.join(
        '/'.join(graph_root.split('/')[:-1]), filename + '_wrong_proteins.txt')
    with open(saved_filename, 'w') as f:
        for file in wrong_proteins:
            f.write(file + '\n')

    return saved_filename_pt

# @functional_transform('normalize_protein')


class NormalizeProtein(BaseTransform):
    r"""Centers and normalizes node positions to the interval :math:`(-1, 1)`
    (functional name: :obj:`normalize_scale`).
    """

    def __init__(self, filename, skip_x=20, skip_edge_attr=64, safe_domi=1e-10):

        dic = torch.load(filename)
        self.skip_x = skip_x
        self.skip_edge_attr = skip_edge_attr
        self.safe_domi = safe_domi
        self.x_mean = dic['x_mean']
        self.x_std = dic['x_std']
        self.pos_mean = dic['pos_mean']
        self.pos_std = torch.mean(dic['pos_std'])
        self.edge_attr_mean = dic['edge_attr_mean']
        self.edge_attr_std = dic['edge_attr_std']

    def __call__(self, data):
        data.x[:, self.skip_x:] = (data.x[:, self.skip_x:] - self.x_mean[self.skip_x:]
                                   ).div_(self.x_std[self.skip_x:] + self.safe_domi)
        data.pos = data.pos - data.pos.mean(dim=-2, keepdim=False)
        data.pos = data.pos.div_(self.pos_std + self.safe_domi)
        data.edge_attr[:, self.skip_edge_attr:] = (data.edge_attr[:, self.skip_edge_attr:]
                                                   - self.edge_attr_mean[self.skip_edge_attr:]).div_(self.edge_attr_std[self.skip_edge_attr:] + self.safe_domi)

        return data


class DihedralGeometryError(Exception):
    pass


class AngleGeometryError(Exception):
    pass


ROUND_ERROR = 1e-14


class Logger(object):
    def __init__(self, logpath, syspart=sys.stdout):
        self.terminal = syspart
        self.log = open(logpath, "a")

    def write(self, message):

        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def log(*args):
    print(f'[{datetime.now()}]', *args)


def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1


def one_hot_res(type_idx, num_residue_type=20):
    rec_feat = [0 for _ in range(num_residue_type)]
    if type_idx < num_residue_type:
        rec_feat[type_idx] = 1
        return rec_feat
    else:
        # print("Warning: residue type index exceeds "+num_residue_type+" !")
        return False


def nan_to_num(ts, val=0.0):
    """
    Replaces nans in tensor with a fixed value.
    """
    val = torch.tensor(val, dtype=ts.dtype, device=ts.device)
    return torch.where(~torch.isfinite(ts), val, ts)


def normalize(tensor, dim=-1):
    """
    Normalizes a tensor along a dimension after removing nans.
    """
    return nan_to_num(
        torch.div(tensor, norm(tensor, dim=dim, keepdim=True))
    )

# def norm(tensor, dim, eps=1e-8, keepdim=False):
#     """
#     Returns L2 norm along a dimension.
#     """
#     return torch.sqrt(
#             torch.sum(torch.square(tensor), dim=dim, keepdim=keepdim) + eps)


def norm(a):
    """Returns the norm of a matrix or vector
    Calculates the Euclidean norm of a vector.
    Applies the Frobenius norm function to a matrix
    (a.k.a. Euclidian matrix norm)
    a = numpy array
    """
    return sqrt(sum((a*a).flat))


def create_vector(vec):
    """Returns a vector as a numpy array."""
    return array([vec[0], vec[1], vec[2]])


def create_vectors(vec1, vec2, vec3, vec4):
    """Returns dihedral angle, takes four
    Scientific.Geometry.Vector objects
    (dihedral does not work for them because
    the Win and Linux libraries are not identical.
    """
    return map(create_vector, [vec1, vec2, vec3, vec4])


def fix_rounding_error(x):
    """If x is almost in the range 0-1, fixes it.
    Specifically, if x is between -ROUND_ERROR and 0, returns 0.
    If x is between 1 and 1+ROUND_ERROR, returns 1.
    """
    if -ROUND_ERROR < x < 0:
        return 0
    elif 1 < x < 1+ROUND_ERROR:
        return 1
    else:
        return


def angle(v1, v2):
    """
    calculates the angle between two vectors.
    v1 and v2 are numpy.array objects.
    returns a float containing the angle in radians.
    """
    length_product = norm(v1)*norm(v2)
    if length_product == 0:
        raise AngleGeometryError(
            "Cannot calculate angle for vectors with length zero")
    cosine = scalar(v1, v2)/length_product
    # angle = arccos(fix_rounding_error(cosine))
    angle = arccos(cosine)

    return angle


def scalar(v1, v2):
    """
    calculates the scalar product of two vectors
    v1 and v2 are numpy.array objects.
    returns a float for a one-dimensional array.
    """
    return sum(v1*v2)


def dihedral(vec1, vec2, vec3, vec4):
    """
    Returns a float value for the dihedral angle between
    the four vectors. They define the bond for which the
    torsion is calculated (~) as:
    V1 - V2 ~ V3 - V4
    The vectors vec1 .. vec4 can be array objects, lists or tuples of length
    three containing floats.
    For Scientific.geometry.Vector objects the behavior is different
    on Windows and Linux. Therefore, the latter is not a featured input type
    even though it may work.
    If the dihedral angle cant be calculated (because vectors are collinear),
    the function raises a DihedralGeometryError
    """
    # create array instances.
    v1, v2, v3, v4 = create_vectors(vec1, vec2, vec3, vec4)
    all_vecs = [v1, v2, v3, v4]

    # rule out that two of the atoms are identical
    # except the first and last, which may be.
    for i in range(len(all_vecs)-1):
        for j in range(i+1, len(all_vecs)):
            if i > 0 or j < 3:  # exclude the (1,4) pair
                equals = all_vecs[i] == all_vecs[j]
                if equals.all():
                    raise DihedralGeometryError(
                        "Vectors #%i and #%i may not be identical!" % (i, j))

    # calculate vectors representing bonds
    v12 = v2-v1
    v23 = v3-v2
    v34 = v4-v3

    # calculate vectors perpendicular to the bonds
    normal1 = cross(v12, v23)
    normal2 = cross(v23, v34)

    # check for linearity
    if norm(normal1) == 0 or norm(normal2) == 0:
        raise DihedralGeometryError(
            "Vectors are in one line; cannot calculate normals!")

    # normalize them to length 1.0
    normal1 = normal1/norm(normal1)
    normal2 = normal2/norm(normal2)

    # calculate torsion and convert to degrees
    torsion = angle(normal1, normal2) * 180.0/pi

    # take into account the determinant
    # (the determinant is a scalar value distinguishing
    # between clockwise and counter-clockwise torsion.
    if scalar(normal1, v34) >= 0:
        return torsion
    else:
        torsion = 360-torsion
        if torsion == 360:
            torsion = 0.0
        return torsion


def seq_dist_distrib(loader):
    before = time()
    q = np.array([0.25, 0.5, 0.75, 0.9, 0.95, 0.98, 0.99])
    seq_dist = torch.Tensor(0)
    for _, data in tqdm(enumerate(loader)):
        edge_attr = data.edge_attr
        seq_dist = torch.cat([seq_dist, edge_attr[:, 0]])
    print('time elapsed: ', time() - before)
    print(np.quantile(np.array(seq_dist), q))
    bins = 200
    sns.distplot(seq_dist.numpy(), hist=True, kde=False, bins=bins)
    plt.hist(seq_dist.numpy(), bins=bins, range=(0, 200))
    plt.title('Histogram of Sequence Distance')
    plt.xlabel('Sequence Distance')
    plt.ylabel('Times')
    plt.savefig('protein/dataset_alpha_Fold/Val_seq_dist' + str(bins) + '.png')
    return seq_dist


def mutat_test4(loader, device, model, dataset):
    # printed cor arranged along the protein names in dataset
    model.eval()
    m = torch.nn.Softmax()
    correct = 0
    protein_names = dataset.protein_names
    n = len(protein_names)

    true_score = [torch.tensor([]).to(device) for _ in range(n)]
    pred_score = [torch.tensor([]).to(device) for _ in range(n)]
    with torch.no_grad():
        # Iterate in batches over the training/test dataset.
        for data in loader:

            ### calculate in model
            data = data.to(device)
            protein_idx = data.protein_idx
            print(protein_idx)
            #data = pre_transform(data)
            x = torch.cat([data.pos, data.mu_r_norm, data.x],
                          dim=1)  # data.mu_r_norm,
            out = model(x, edge_index=data.edge_index,
                        edge_attr=data.edge_attr, batch=data.batch)
            out = torch.log(m(out[:, :20]))
            # out = out.cpu()

            # obtain score info
            score_info = data.score_info[0]
            num_mutat = len(score_info)
            true_score[protein_idx] = torch.zeros(num_mutat)
            pred_score[protein_idx] = torch.zeros(num_mutat)
            for mutat_idx in range(num_mutat):
                mutat_info, true_score[protein_idx][mutat_idx] = score_info[mutat_idx]
                for item in mutat_info:
                    # (log(prob_out) - log(prob_wlid))
                    # print(protein_idx,mutat_idx)
                    if int(item[1]) >= out.shape[0]:
                        continue
                    pred_score[protein_idx][mutat_idx] += (
                        out[int(item[1]), int(item[2])] - out[int(item[1]), int(item[0])]).cpu()
    spearman_coeef = np.zeros(n)
    for i in range(n):
        if len(true_score[i].cpu().numpy()) == 0:
            continue
        spearvalue = spearmanr(true_score[i].cpu().numpy(
        ), pred_score[i].cpu().numpy()).correlation
        if spearvalue is nan:
            pass
        else:
            spearman_coeef[i] = spearvalue
    for i in range(len(spearman_coeef)):
        print(protein_names[i])
        print(spearman_coeef[i])
    # Derive ratio of correct predictions.
    return correct / len(loader.dataset), spearman_coeef.mean()


def normalize_prob(coeff):
      sum_ = coeff.sum(dim = 1)
      score = coeff/(sum_.view(20,-1))
      return score

def substitute_label(y,temperature=1.0):
    original_score = torch.tensor([
        [ 8.,  3.,  2.,  2.,  4.,  3.,  3.,  4.,  2.,  3.,  3.,  3.,  3.,  2.,
          3.,  5.,  4.,  1.,  2.,  4.],
        [ 3.,  9.,  4.,  2.,  1.,  5.,  4.,  2.,  4.,  1.,  2.,  6.,  3.,  1.,
          2.,  3.,  3.,  1.,  2.,  1.],
        [ 2.,  4., 10.,  5.,  1.,  4.,  4.,  4.,  5.,  1.,  1.,  4.,  2.,  1.,
          2.,  5.,  4.,  0.,  2.,  1.],
        [ 2.,  2.,  5., 10.,  1.,  4.,  6.,  3.,  3.,  1.,  0.,  3.,  1.,  1.,
          3.,  4.,  3.,  0.,  1.,  1.],
        [ 4.,  1.,  1.,  1., 13.,  1.,  0.,  1.,  1.,  3.,  3.,  1.,  3.,  2.,
          1.,  3.,  3.,  2.,  2.,  3.],
        [ 3.,  5.,  4.,  4.,  1.,  9.,  6.,  2.,  4.,  1.,  2.,  5.,  4.,  1.,
          3.,  4.,  3.,  2.,  3.,  2.],
        [ 3.,  4.,  4.,  6.,  0.,  6.,  9.,  2.,  4.,  1.,  1.,  5.,  2.,  1.,
          3.,  4.,  3.,  1.,  2.,  2.],
        [ 4.,  2.,  4.,  3.,  1.,  2.,  2., 10.,  2.,  0.,  0.,  2.,  1.,  1.,
          2.,  4.,  2.,  2.,  1.,  1.],
        [ 2.,  4.,  5.,  3.,  1.,  4.,  4.,  2., 12.,  1.,  1.,  3.,  2.,  3.,
          2.,  3.,  2.,  2.,  6.,  1.],
        [ 3.,  1.,  1.,  1.,  3.,  1.,  1.,  0.,  1.,  8.,  6.,  1.,  5.,  4.,
          1.,  2.,  3.,  1.,  3.,  7.],
        [ 3.,  2.,  1.,  0.,  3.,  2.,  1.,  0.,  1.,  6.,  8.,  2.,  6.,  4.,
          1.,  2.,  3.,  2.,  3.,  5.],
        [ 3.,  6.,  4.,  3.,  1.,  5.,  5.,  2.,  3.,  1.,  2.,  9.,  3.,  1.,
          3.,  4.,  3.,  1.,  2.,  2.],
        [ 3.,  3.,  2.,  1.,  3.,  4.,  2.,  1.,  2.,  5.,  6.,  3.,  9.,  4.,
          2.,  3.,  3.,  3.,  3.,  5.],
        [ 2.,  1.,  1.,  1.,  2.,  1.,  1.,  1.,  3.,  4.,  4.,  1.,  4., 10.,
          0.,  2.,  2.,  5.,  7.,  3.],
        [ 3.,  2.,  2.,  3.,  1.,  3.,  3.,  2.,  2.,  1.,  1.,  3.,  2.,  0.,
         11.,  3.,  3.,  0.,  1.,  2.],
        [ 5.,  3.,  5.,  4.,  3.,  4.,  4.,  4.,  3.,  2.,  2.,  4.,  3.,  2.,
          3.,  8.,  5.,  1.,  2.,  2.],
        [ 4.,  3.,  4.,  3.,  3.,  3.,  3.,  2.,  2.,  3.,  3.,  3.,  3.,  2.,
          3.,  5.,  9.,  2.,  2.,  4.],
        [ 1.,  1.,  0.,  0.,  2.,  2.,  1.,  2.,  2.,  1.,  2.,  1.,  3.,  5.,
          0.,  1.,  2., 15.,  6.,  1.],
        [ 2.,  2.,  2.,  1.,  2.,  3.,  2.,  1.,  6.,  3.,  3.,  2.,  3.,  7.,
          1.,  2.,  2.,  6., 11.,  3.],
        [ 4.,  1.,  1.,  1.,  3.,  2.,  2.,  1.,  1.,  7.,  5.,  2.,  5.,  3.,
          2.,  2.,  4.,  1.,  3.,  8.]], dtype=torch.float64,device=y.device)
    
    score = normalize_prob(original_score)
    tempering_score = score**temperature
    normalized_score = normalize_prob(tempering_score)
    out_prob = normalized_score[y]
  
    return out_prob

def substitution_label_smooth(y):  # [1,4,5,6,7]
    '''
    mapping tensor convert dataset label    ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    to substitution tabel label     ['C', 'S',	'T','A','G','P'	,'D','E',	'Q'	,'N',	'H','R','K','M'	,'I','L','V','W','Y','F']
    '''

    score = torch.tensor([[8.4015e-01, 5.6609e-03, 2.0825e-03, 2.0825e-03, 1.5388e-02, 5.6609e-03,
                           5.6609e-03, 1.5388e-02, 2.0825e-03, 5.6609e-03, 5.6609e-03, 5.6609e-03,
                           5.6609e-03, 2.0825e-03, 5.6609e-03, 4.1829e-02, 1.5388e-02, 7.6612e-04,
                           2.0825e-03, 1.5388e-02],
                          [2.2443e-03, 9.0541e-01, 6.1006e-03, 8.2563e-04, 3.0373e-04, 1.6583e-02,
                           6.1006e-03, 8.2563e-04, 6.1006e-03, 3.0373e-04, 8.2563e-04, 4.5078e-02,
                           2.2443e-03, 3.0373e-04, 8.2563e-04, 2.2443e-03, 2.2443e-03, 3.0373e-04,
                           8.2563e-04, 3.0373e-04],
                          [3.2347e-04, 2.3901e-03, 9.6424e-01, 6.4970e-03, 1.1900e-04, 2.3901e-03,
                           2.3901e-03, 2.3901e-03, 6.4970e-03, 1.1900e-04, 1.1900e-04, 2.3901e-03,
                           3.2347e-04, 1.1900e-04, 3.2347e-04, 6.4970e-03, 2.3901e-03, 4.3776e-05,
                           3.2347e-04, 1.1900e-04],
                          [3.2378e-04, 3.2378e-04, 6.5034e-03, 9.6518e-01, 1.1911e-04, 2.3925e-03,
                           1.7678e-02, 8.8013e-04, 8.8013e-04, 1.1911e-04, 4.3819e-05, 8.8013e-04,
                           1.1911e-04, 1.1911e-04, 8.8013e-04, 2.3925e-03, 8.8013e-04, 4.3819e-05,
                           1.1911e-04, 1.1911e-04],
                          [1.2335e-04, 6.1412e-06, 6.1412e-06, 6.1412e-06, 9.9950e-01, 6.1412e-06,
                           2.2592e-06, 6.1412e-06, 6.1412e-06, 4.5377e-05, 4.5377e-05, 6.1412e-06,
                           4.5377e-05, 1.6693e-05, 6.1412e-06, 4.5377e-05, 4.5377e-05, 1.6693e-05,
                           1.6693e-05, 4.5377e-05],
                          [2.1845e-03, 1.6142e-02, 5.9382e-03, 5.9382e-03, 2.9565e-04, 8.8131e-01,
                           4.3878e-02, 8.0365e-04, 5.9382e-03, 2.9565e-04, 8.0365e-04, 1.6142e-02,
                           5.9382e-03, 2.9565e-04, 2.1845e-03, 5.9382e-03, 2.1845e-03, 8.0365e-04,
                           2.1845e-03, 8.0365e-04],
                          [2.1417e-03, 5.8217e-03, 5.8217e-03, 4.3017e-02, 1.0663e-04, 4.3017e-02,
                           8.6401e-01, 7.8788e-04, 5.8217e-03, 2.8984e-04, 2.8984e-04, 1.5825e-02,
                           7.8788e-04, 2.8984e-04, 2.1417e-03, 5.8217e-03, 2.1417e-03, 2.8984e-04,
                           7.8788e-04, 7.8788e-04],
                          [2.4500e-03, 3.3157e-04, 2.4500e-03, 9.0130e-04, 1.2198e-04, 3.3157e-04,
                           3.3157e-04, 9.8840e-01, 3.3157e-04, 4.4873e-05, 4.4873e-05, 3.3157e-04,
                           1.2198e-04, 1.2198e-04, 3.3157e-04, 2.4500e-03, 3.3157e-04, 3.3157e-04,
                           1.2198e-04, 1.2198e-04],
                          [4.5164e-05, 3.3372e-04, 9.0714e-04, 1.2277e-04, 1.6615e-05, 3.3372e-04,
                           3.3372e-04, 4.5164e-05, 9.9480e-01, 1.6615e-05, 1.6615e-05, 1.2277e-04,
                           4.5164e-05, 1.2277e-04, 4.5164e-05, 1.2277e-04, 4.5164e-05, 4.5164e-05,
                           2.4659e-03, 1.6615e-05],
                          [4.1869e-03, 5.6664e-04, 5.6664e-04, 5.6664e-04, 4.1869e-03, 5.6664e-04,
                           5.6664e-04, 2.0845e-04, 5.6664e-04, 6.2139e-01, 8.4096e-02, 5.6664e-04,
                           3.0937e-02, 1.1381e-02, 5.6664e-04, 1.5403e-03, 4.1869e-03, 5.6664e-04,
                           4.1869e-03, 2.2860e-01],
                          [4.8740e-03, 1.7930e-03, 6.5962e-04, 2.4266e-04, 4.8740e-03, 1.7930e-03,
                           6.5962e-04, 2.4266e-04, 6.5962e-04, 9.7896e-02, 7.2336e-01, 1.7930e-03,
                           9.7896e-02, 1.3249e-02, 6.5962e-04, 1.7930e-03, 4.8740e-03, 1.7930e-03,
                           4.8740e-03, 3.6014e-02],
                          [2.2137e-03, 4.4462e-02, 6.0173e-03, 2.2137e-03, 2.9959e-04, 1.6357e-02,
                           1.6357e-02, 8.1436e-04, 2.2137e-03, 2.9959e-04, 8.1436e-04, 8.9305e-01,
                           2.2137e-03, 2.9959e-04, 2.2137e-03, 6.0173e-03, 2.2137e-03, 2.9959e-04,
                           8.1436e-04, 8.1436e-04],
                          [2.2052e-03, 2.2052e-03, 8.1125e-04, 2.9844e-04, 2.2052e-03, 5.9944e-03,
                           8.1125e-04, 2.9844e-04, 8.1125e-04, 1.6294e-02, 4.4293e-02, 2.2052e-03,
                           8.8965e-01, 5.9944e-03, 8.1125e-04, 2.2052e-03, 2.2052e-03, 2.2052e-03,
                           2.2052e-03, 1.6294e-02],
                          [3.1409e-04, 1.1555e-04, 1.1555e-04, 1.1555e-04, 3.1409e-04, 1.1555e-04,
                           1.1555e-04, 1.1555e-04, 8.5379e-04, 2.3209e-03, 2.3209e-03, 1.1555e-04,
                           2.3209e-03, 9.3630e-01, 4.2508e-05, 3.1409e-04, 3.1409e-04, 6.3087e-03,
                           4.6616e-02, 8.5379e-04],
                          [3.3436e-04, 1.2300e-04, 1.2300e-04, 3.3436e-04, 4.5250e-05, 3.3436e-04,
                           3.3436e-04, 1.2300e-04, 1.2300e-04, 4.5250e-05, 4.5250e-05, 3.3436e-04,
                           1.2300e-04, 1.6647e-05, 9.9671e-01, 3.3436e-04, 3.3436e-04, 1.6647e-05,
                           4.5250e-05, 1.2300e-04],
                          [3.8657e-02, 5.2316e-03, 3.8657e-02, 1.4221e-02, 5.2316e-03, 1.4221e-02,
                           1.4221e-02, 1.4221e-02, 5.2316e-03, 1.9246e-03, 1.9246e-03, 1.4221e-02,
                           5.2316e-03, 1.9246e-03, 5.2316e-03, 7.7644e-01, 3.8657e-02, 7.0802e-04,
                           1.9246e-03, 1.9246e-03],
                          [6.3097e-03, 2.3212e-03, 6.3097e-03, 2.3212e-03, 2.3212e-03, 2.3212e-03,
                           2.3212e-03, 8.5392e-04, 8.5392e-04, 2.3212e-03, 2.3212e-03, 2.3212e-03,
                           2.3212e-03, 8.5392e-04, 2.3212e-03, 1.7151e-02, 9.3644e-01, 8.5392e-04,
                           8.5392e-04, 6.3097e-03],
                          [8.3137e-07, 8.3137e-07, 3.0584e-07, 3.0584e-07, 2.2599e-06, 2.2599e-06,
                           8.3137e-07, 2.2599e-06, 2.2599e-06, 8.3137e-07, 2.2599e-06, 8.3137e-07,
                           6.1430e-06, 4.5391e-05, 3.0584e-07, 8.3137e-07, 2.2599e-06, 9.9980e-01,
                           1.2339e-04, 8.3137e-07],
                          [1.1928e-04, 1.1928e-04, 1.1928e-04, 4.3882e-05, 1.1928e-04, 3.2425e-04,
                           1.1928e-04, 4.3882e-05, 6.5127e-03, 3.2425e-04, 3.2425e-04, 1.1928e-04,
                           3.2425e-04, 1.7703e-02, 4.3882e-05, 1.1928e-04, 1.1928e-04, 6.5127e-03,
                           9.6656e-01, 3.2425e-04],
                          [1.1877e-02, 5.9130e-04, 5.9130e-04, 5.9130e-04, 4.3692e-03, 1.6073e-03,
                           1.6073e-03, 5.9130e-04, 5.9130e-04, 2.3855e-01, 3.2284e-02, 1.6073e-03,
                           3.2284e-02, 4.3692e-03, 1.6073e-03, 1.6073e-03, 1.1877e-02, 5.9130e-04,
                           4.3692e-03, 6.4844e-01]], dtype=torch.float64, device=y.device)
    out_prob = score[y]

    return out_prob  # []