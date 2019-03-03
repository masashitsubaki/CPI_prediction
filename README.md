# Compound-protein interaction (CPI) prediction using a GNN for compounds and a CNN for proteins

This code is an implementation of our paper
"[Compound-protein Interaction Prediction with End-to-end Learning of Neural Networks for Graphs and Sequences (Bioinformatics, 2018)](https://academic.oup.com/bioinformatics/advance-article-abstract/doi/10.1093/bioinformatics/bty535/5050020?redirectedFrom=PDF)" in PyTorch.
In this repository, we provide two CPI datasets: human and *C. elegans* created by
"[Improving compound–protein interaction prediction by building up highly credible negative samples (Bioinformatics, 2015).](https://academic.oup.com/bioinformatics/article/31/12/i221/216307)"
Note that the ratio of positive and negative samples is 1:1.

In our problem setting of CPI prediction,
an input is the pair of a SMILES format of compound and an amino acid sequence of protein;
an output is a binary label (interact or not).
The SMILES is converted with RDKit and
we obtain a 2D graph-structured data of the compound (i.e., atom types and their adjacency matrix).
The overview of our **CPI prediction by GNN-CNN** is as follows:

<div align="center">
<p><img src="model.jpeg" width="600" /></p>
</div>

The details of the GNN and CNN are described in our paper.

Note that, the above CPI prediction uses our proposed GNN,
which is based on learning representations of r-radius subgraphs (i.e., fingerprints) in molecules.
We also provide an implementation of the GNN for predicting various molecular properties
such as drug efficacy and photovoltaic efficiency in https://github.com/masashitsubaki/GNN_molecules.


## Characteristics

- This code is easy to use. After setting the environment (e.g., PyTorch),
preprocessing data and learning a model can be done by only two commands (see "Usage").
- If you prepare a CPI dataset with the same format as provided in the dataset directory,
you can learn our GNN-CNN with your dataset by the two commands
(see "Training of our GNN-CNN using your CPI dataset").


## Requirements

- PyTorch
- scikit-learn
- RDKit


## Usage

We provide two major scripts:

- code/preprocess_data.py creates the input tensor data of CPIs
for processing with PyTorch from the original data
(see dataset/human or celegans/original/data.txt).
- code/run_training.py trains the model using the above preprocessed data
(see dataset/human or celegans/input).

(i) Create the tensor data of CPIs with the following command:
```
cd code
bash preprocess_data.sh
```

The preprocessed data are saved in the dataset/input directory.

(ii) Using the preprocessed data, train the model with the following command:
```
bash run_training.sh
```

The training and test results and the model are saved in the output directory
(after training, see output/result and output/model).

(iii) You can change the hyperparameters in preprocess_data.sh and run_training.sh.
Try to learn various models.


## Result

Learning curves (x-axis is epoch and y-axis is AUC)
on the test datasets of human and *C. elegans* are as follows:

<div align="center">
<p><img src="learning_curves.jpeg" width="800" /></p>
</div>

These results can be reproduce by the above two commands (i) and (ii).


## Training of our GNN-CNN using your CPI dataset
In the directory of dataset/human or celegans/original,
we now have the original data "data.txt" as follows:

```
CC[C@@]...OC)O MSPLNQ...KAS 0
C1C...O1 MSTSSL...FLL 1
CCCC(=O)...CC=C1 MAGAGP...QET 0
...
...
...
CC...C MKGNST...FVS 0
C(C...O)N MSPSPT...LCS 1
```

Each line has "SMILES sequence interaction."
Note that, the interaction 1 means that "the pair of SMILES and sequence has interaction" and
0 means that "the pair does not have interaction."
If you prepare a dataset with the same format as "data.txt" in a new directory
(e.g., dataset/yourdata/original),
you can train our GNN-CNN using your dataset by the above two commands (i) and (ii).


## TODO

- Preprocess data contains "." in the SMILES format (i.e., a molecule contains multi-graphs).
- Provide some pre-trained model and the demo scripts.
- Implement an efficient batch processing of the attention mechanism
bridging two different architectures (GNN and CNN).


## How to cite

```
@article{tsubaki2018compound,
  title={Compound-protein Interaction Prediction with End-to-end Learning of Neural Networks for Graphs and Sequences},
  author={Tsubaki, Masashi and Tomii, Kentaro and Sese, Jun},
  journal={Bioinformatics},
  year={2018}
}
```
