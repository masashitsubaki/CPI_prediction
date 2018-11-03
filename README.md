# Compound-protein Interaction Prediction

This code is a simpler model and its faster implementation of
"[Compound-protein Interaction Prediction with End-to-end Learning of Neural Networks for Graphs and Sequences (Bioinformatics, 2018)](https://academic.oup.com/bioinformatics/advance-article-abstract/doi/10.1093/bioinformatics/bty535/5050020?redirectedFrom=PDF)" in PyTorch.


## Requirements

- PyTorch (version 0.4.0)
- scikit-learn
- RDKit


## Usage

We provide two major scripts:

- code/preprocess_data.py creates the input tensor data of compound-protein interactions (CPIs) for processing with PyTorch from the original data (see dataset/human/original/smiles_sequence_interaction.txt).
- code/run_training.py trains our neural network using the above preprocessed data (see dataset/human/input) to predict CPIs.

(i) Create the tensor data of CPIs with the following command:
```
cd code
bash preprocess_data.sh
```

(ii) Using the preprocessed data, train our neural network with the following command:
```
bash run_training.sh
```

The training result and trained model are saved in the output directory (after training, see output/result and output/model).

(iii) You can change the hyperparameters in preprocess_data.sh and run_training.sh, and try to learn various models!


## Train our neural network using your CPI dataset
In the directory of dataset/human/original, we now have "smiles_sequence_interaction.txt." If you prepare dataset with the same format as "smiles_sequence_interaction.txt" in a new directory (e.g., dataset/yourdata/original), you can train our neural network using your dataset by the above tow commands (i) and (ii).


## Future work

- Provide a pre-trained model with a large dataset.
- Provide a code for analyzing 3D interaction sites using obtained attention weights.
