# Compound-protein Interaction (CPI) Prediction on the human dataset

This code is a simpler model and its faster implementation of our paper
"[Compound-protein Interaction Prediction with End-to-end Learning of Neural Networks for Graphs and Sequences (Bioinformatics, 2018)](https://academic.oup.com/bioinformatics/advance-article-abstract/doi/10.1093/bioinformatics/bty535/5050020?redirectedFrom=PDF)" in PyTorch.
In this code, we use the CPI dataset of human provided in
"[Improving compound–protein interaction prediction by building up highly credible negative samples (Bioinformatics, 2015).](https://academic.oup.com/bioinformatics/article/31/12/i221/216307)"
Note that the ratio of positive and negative samples is 1:1.

In the problem of CPI prediction,
an input is the pair of a SMILES (compound) and an amino acid sequence (protein);
an ouput is a binary label (interact or not).
The SMILES is converted with RDKit and
we obtain a graph of the compound (i.e., atom types and their adjacent matrix).
The overviwe of our CPI prediction approach is as follows:

<div align="center">
<p><img src="model.jpeg" width="500" /></p>
</div>

The details of the above model are described in our paper.

Note that, in our paper we propose a graph neural network (GNN) for molecules,
which is based on learning representations of
r-radius subgraphs (or called fingerprints) in molecules.
The details of our GNN and its implementation for predicting various molecular properties
are provided in https://github.com/masashitsubaki/GNN_molecules.


## Characteristics

- This code is easy to use. After setting the environment (e.g., PyTorch),
preprocessing data and learning a model can be done by only two commands (see "Usage").
- If you prepare dataset with the same format as provided in the dataset directory,
your can learn our model with your dataset by the two commands
(see "Training of our neural network using your CPI dataset").


## Requirements

- PyTorch (version 0.4.0)
- scikit-learn
- RDKit


## Usage

We provide two major scripts:

- code/preprocess_data.py creates the input tensor data of compound-protein interactions (CPIs)
for processing with PyTorch from the original data (see dataset/human/original/smiles_sequence_interaction.txt).
- code/run_training.py trains our neural network
using the above preprocessed data (see dataset/human/input) to predict CPIs.

(i) Create the tensor data of CPIs with the following command:
```
cd code
bash preprocess_data.sh
```

(ii) Using the preprocessed data, train our neural network with the following command:
```
bash run_training.sh
```

The training result and trained model are saved in the output directory
(after training, see output/result and output/model).

(iii) You can change the hyperparameters in preprocess_data.sh and run_training.sh. Try to learn various models!


## Training of our neural network using your CPI dataset
In the directory of dataset/human/original, we now have "smiles_sequence_interaction.txt." If you prepare dataset with the same format as "smiles_sequence_interaction.txt" in a new directory (e.g., dataset/yourdata/original), you can train our neural network using your dataset by the above two commands (i) and (ii).


## TODO

- Provide a pre-trained model with a large dataset.


## How to cite

```
@article{tsubaki2018compound,
  title={Compound-protein Interaction Prediction with End-to-end Learning of Neural Networks for Graphs and Sequences},
  author={Tsubaki, Masashi and Tomii, Kentaro and Sese, Jun},
  journal={Bioinformatics},
  year={2018}
}
```
