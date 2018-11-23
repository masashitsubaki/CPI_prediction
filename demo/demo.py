from collections import defaultdict
import pickle
import sys
import timeit

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem


def create_atoms(mol):
    atoms = [atom_dict[a.GetSymbol()] for a in mol.GetAtoms()]
    return np.array(atoms)


def create_ijbonddict(mol):
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))
    return i_jbond_dict


def create_fingerprints(atoms, i_jbond_dict, radius):
    if (len(atoms) == 1) or (radius == 0):
        fingerprints = [fingerprint_dict[a] for a in atoms]

    else:
        vertices = atoms
        for _ in range(radius):
            fingerprints = []
            for i, j_bond in i_jbond_dict.items():
                neighbors = [(vertices[j], bond) for j, bond in j_bond]
                fingerprint = (vertices[i], tuple(sorted(neighbors)))
                fingerprints.append(fingerprint_dict[fingerprint])
            vertices = fingerprints

    return np.array(fingerprints)


def create_adjacency(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency)


def split_sequence(sequence, ngram):
    sequence = '-' + sequence + '='
    words = [word_dict[sequence[i:i+ngram]]
             for i in range(len(sequence)-ngram+1)]
    return np.array(words)


class CompoundProteinInteractionPrediction(nn.Module):
    def __init__(self):
        super(CompoundProteinInteractionPrediction, self).__init__()
        self.embed_fingerprint = nn.Embedding(n_fingerprint, dim)
        self.embed_word = nn.Embedding(n_word, dim)
        self.W_gnn = nn.ModuleList([nn.Linear(dim, dim)
                                    for _ in range(layer_gnn)])
        self.W_cnn = nn.ModuleList([nn.Conv2d(
                     in_channels=1, out_channels=1, kernel_size=2*window+1,
                     stride=1, padding=window) for _ in range(layer_cnn)])
        self.W_attention = nn.Linear(dim, dim)
        self.W_out = nn.Linear(2*dim, 2)

    def gnn(self, xs, A, layer):
        for i in range(layer):
            hs = torch.relu(self.W_gnn[i](xs))
            xs = xs + torch.matmul(A, hs)
        return torch.unsqueeze(torch.sum(xs, 0), 0)

    def cnn(self, xs, i):
        xs = torch.unsqueeze(torch.unsqueeze(xs, 0), 0)
        hs = torch.relu(self.W_cnn[i](xs))
        return torch.squeeze(torch.squeeze(hs, 0), 0)

    def attention_cnn(self, x, xs, layer):
        for i in range(layer):
            hs = self.cnn(xs, i)
            x = torch.relu(self.W_attention(x))
            hs = torch.relu(self.W_attention(hs))
            weights = torch.tanh(F.linear(x, hs))
            xs = torch.t(weights) * hs
        return torch.unsqueeze(torch.sum(xs, 0), 0)

    def __call__(self, data):

        fingerprints, adjacency, words = data

        x_fingerprints = self.embed_fingerprint(fingerprints)
        x_compound = self.gnn(x_fingerprints, adjacency, layer_gnn)

        x_words = self.embed_word(words)
        x_protein = self.attention_cnn(x_compound, x_words, layer_cnn)

        y_cat = torch.cat((x_compound, x_protein), 1)
        z_interaction = self.W_out(y_cat)
        z = F.softmax(z_interaction, 1).to('cpu').data[0].numpy()

        return z


class Predictor(object):
    def __init__(self, model):
        """Load the pre-trained model from the directory of output/model."""
        self.model = model
        model.load_state_dict(torch.load('../output/model/' + setting))

    def predict(self, dataset, smiles_sequence_list):

        z_list, t_list = [], []
        for data in dataset:
            z = self.model(data)
            z_list.append(z[1])
            t_list.append(np.argmax(z))

        with open('prediction_result.txt', 'w') as f:
            f.write('smiles sequence '
                    'interaction_probability binary_class\n')
            for (c, p), z, t in zip(smiles_sequence_list, z_list, t_list):
                f.write(' '.join(map(str, [c, p, z, t])) + '\n')


def load_dictionary(file_name):
    with open(file_name, 'rb') as f:
        d = pickle.load(f)
    dictionary = defaultdict(lambda: len(d))
    dictionary.update(d)
    return dictionary


if __name__ == "__main__":

    (DATASET, radius, ngram, dim, layer_gnn, window, layer_cnn,
     setting) = sys.argv[1:]

    (dim, layer_gnn, window,
     layer_cnn) = map(int, [dim, layer_gnn, window, layer_cnn])

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    print('Loading data...')

    with open('smiles_sequence.txt', 'r') as f:
        cp_list = f.read().strip().split('\n')
    cp_list = list(filter(lambda x:
                   '.' not in x.strip().split()[0], cp_list))

    dir_input = ('../dataset/' + DATASET + '/input/'
                 'radius' + radius + '_ngram' + ngram + '/')

    atom_dict = load_dictionary(dir_input + 'atom_dict.pickle')
    bond_dict = load_dictionary(dir_input + 'bond_dict.pickle')
    fingerprint_dict = load_dictionary(dir_input + 'fingerprint_dict.pickle')
    word_dict = load_dictionary(dir_input + 'word_dict.pickle')
    n_fingerprint = len(fingerprint_dict) + 1
    n_word = len(word_dict) + 1

    radius, ngram = map(int, [radius, ngram])

    print('Creating data...')

    Compounds, Adjacencies, Proteins, Interactions = [], [], [], []
    smiles_sequence_list = []

    for cp in cp_list:

        smiles, sequence = cp.strip().split()
        smiles_sequence_list.append((smiles, sequence))

        mol = Chem.MolFromSmiles(smiles)
        atoms = create_atoms(mol)
        i_jbond_dict = create_ijbonddict(mol)

        fingerprints = create_fingerprints(atoms, i_jbond_dict, radius)
        Compounds.append(torch.LongTensor(fingerprints).to(device))

        adjacency = create_adjacency(mol)
        Adjacencies.append(torch.FloatTensor(adjacency).to(device))

        words = split_sequence(sequence, ngram)
        Proteins.append(torch.LongTensor(words).to(device))

    dataset = list(zip(Compounds, Adjacencies, Proteins))

    print('Predictiing CPI...')

    model = CompoundProteinInteractionPrediction().to(device)
    predictor = Predictor(model)

    start = timeit.default_timer()
    predictor.predict(dataset, smiles_sequence_list)
    end = timeit.default_timer()
    time = end - start

    print('Prediction has finished in ' + str(time) + ' sec!')
