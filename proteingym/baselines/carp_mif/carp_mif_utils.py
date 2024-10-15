import torch
from sequence_models.collaters import SimpleCollater, StructureCollater, BGCCollater
from sequence_models.pretrained import load_carp,load_gnn,MIF
from sequence_models.constants import PROTEIN_ALPHABET

CARP_URL = 'https://zenodo.org/record/6564798/files/'
MIF_URL = 'https://zenodo.org/record/6573779/files/'
BIG_URL = 'https://zenodo.org/record/6857704/files/'

def load_model_and_alphabet(model_name, model_dir=None):
    if not model_name.endswith(".pt"): 
        if 'big' in model_name:
            url = BIG_URL + '%s.pt?download=1' %model_name
        elif 'carp' in model_name:
            url = CARP_URL + '%s.pt?download=1' %model_name
        elif 'mif' in model_name:
            url = MIF_URL + '%s.pt?download=1' %model_name
        model_data = torch.hub.load_state_dict_from_url(url, progress=False, map_location="cpu", model_dir=model_dir)
    else:
        model_data = torch.load(model_name, map_location="cpu")
    if 'big' in model_data['model']:
        pfam_to_domain = model_data['pfam_to_domain']
        tokens = model_data['tokens']
        collater = BGCCollater(tokens, pfam_to_domain)
    else:
        collater = SimpleCollater(PROTEIN_ALPHABET, pad=True)
    if 'carp' in model_data['model']:
        model = load_carp(model_data)
    elif model_data['model'] in ['mif', 'mif-st']:
        gnn = load_gnn(model_data)
        cnn = None
        if model_data['model'] == 'mif-st':
            url = CARP_URL + '%s.pt?download=1' % 'carp_640M'
            cnn_data = torch.hub.load_state_dict_from_url(url, progress=False, map_location="cpu")
            cnn = load_carp(cnn_data)
        collater = StructureCollater(collater, n_connections=30)
        model = MIF(gnn, cnn=cnn)
    return model, collater