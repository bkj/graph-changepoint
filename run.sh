#!/bin/bash

# run.sh

mkdir -p _results/synthetic

# ----------------------------------------------------------------------
# Small synthetic datasets

# Single change point
python mst.py \
    --inpath _data/synthetic/X.tsv \
    --outpath _results/synthetic/X_edges.tsv

python main.py \
    --inpath _results/synthetic/X_edges.tsv \
    --outpath _results/synthetic/X_result.tsv

# Changed interval
python mst.py \
    --inpath _data/synthetic/X2.tsv \
    --outpath _results/synthetic/X2_edges.tsv

python main.py \
    --inpath _results/synthetic/X2_edges.tsv \
    --outpath _results/synthetic/X2_result.tsv \
    --two-d

# ----------------------------------------------------------------------
# Larger real datasets

mkdir -p _results/yankee
python mst.py \
    --inpath _data/yankee/yankee.tsv \
    --outpath _results/yankee/yankee_edges.tsv

python main.py \
    --inpath _results/yankee/yankee_edges.tsv \
    --outpath _results/yankee/yankee_result.tsv

python main.py \
    --inpath _results/yankee/yankee_edges.tsv \
    --outpath _results/yankee/yankee_result.tsv \
    --two-d

# ** In progress **