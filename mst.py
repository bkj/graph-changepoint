#!/usr/bin/env python

"""
    mst.py
    
    !! Helper function for computing euclidean MST on dense data
    !! ** We do not care about benchmarking this **
"""

from __future__ import print_function

import sys
import argparse
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform, pdist
from scipy.sparse.csgraph import minimum_spanning_tree


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', type=str, default="_data/synthetic/X.tsv")
    parser.add_argument('--outpath', type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    print('main.py: loading data', file=sys.stderr)
    X = pd.read_csv(args.inpath, header=None, sep='\t').values
    
    print('main.py: computing MST', file=sys.stderr)
    X_dist = squareform(pdist(X))
    mst    = minimum_spanning_tree(X_dist).tocoo()
    edges  = np.column_stack([mst.row, mst.col])
    pd.DataFrame(edges).to_csv(args.outpath, sep='\t', header=None, index=False)