# Compound Protein Interaction Prediction

This code is a simpler and faster implementation of
[Compound-protein Interaction Prediction with End-to-end Learning of Neural Networks for Graphs and Sequences (Bioinformatics, 2018)](https://academic.oup.com/bioinformatics/advance-article-abstract/doi/10.1093/bioinformatics/bty535/5050020?redirectedFrom=PDF).


## Requirements

The code requires:
- PyTorch (version 0.4.0)
- scikit-learn
- RDKit


## Usage

We provides two major scripts:

- code/preprocess_data.py creates the input tensor data of compound-protein interactions (CPIs)
for processing with PyTorch from the original data
(see dataset/human/original/smiles_sequence_interaction.txt).
- code/run_training.py trains our neural network to predict CPIs.

(i) Create the tensor data of CPIs with the following command:
```
cd code
bash preprocess_data.sh
```

(ii) On the preprocessed data (see dataset/human/input),
train our neural network with the following command:
```
bash run_training.sh
```

The training results and model are saved in the output directory.

(iii) You can change the hyperparameters in preprocess_data.sh and run_training.sh,
and try to learn various models!


## Train your CPI data
In the directory of dataset/human/original, we now have "smiles_sequence_interaction.txt."
If you prepare other data with the same format as "smiles_sequence_interaction.txt"
in a new directory (e.g., dataset/yourdata/original),
you can train your neural network using your data by the commands (i) and (ii).


## Future work

- Provide a pre-trained model with a large dataset.
