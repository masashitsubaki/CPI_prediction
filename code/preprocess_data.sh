#!/bin/bash

DATASET=human
# DATASET=yourdata

radius=2  # >=0.

ngram=3  # >=1.

python preprocess_data.py $DATASET $radius $ngram
