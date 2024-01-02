import torch
from torch import sin, cos, atan2, acos

def rot_z(gamma):
    return torch.tensor([
        [cos(gamma), -sin(gamma), 0],
        [sin(gamma), cos(gamma), 0],
        [0, 0, 1]
    ], dtype=gamma.dtype)

def rot_y(beta):
    return torch.tensor([
        [cos(beta), 0, sin(beta)],
        [0, 1, 0],
        [-sin(beta), 0, cos(beta)]
    ], dtype=beta.dtype)

def rot(alpha, beta, gamma):
    return rot_z(alpha) @ rot_y(beta) @ rot_z(gamma)

def get_node_feature_dims():
    '''
    each node has 25 dim feature corrsponding to residual type, sasa, dihedral, mu_r_norm
    '''
    return [20, 1,1, 4, 5,640]


def get_edge_feature_dims():
    '''
    each node has 93 dim feature corrsponding to one hot sequence distance, interatomic distance, local frame orientation
    '''
    return [65, 1, 15, 12]