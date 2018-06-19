#!/usr/bin/env python
"""
    main.py
    
    Implementation of "Graph-Based Change Point Detection"
        https://arxiv.org/abs/1209.1625.pdf
"""

from __future__ import print_function, division

import argparse
import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial.distance import squareform, pdist
from scipy.sparse.csgraph import minimum_spanning_tree

# --
# Helpers

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', type=str, default='X.tsv')
    parser.add_argument('--outpath', type=str, default='X_rz.tsv')
    return parser.parse_args()


def scan_stat(g):
    R, cross = [], []
    for n in g.nodes:
        for e in g.edges(n):
            if e[1] > n:
                cross.append(e)
        
        cross = [c for c in cross if c[1] > n]
        R.append(len(cross))
    
    return np.array(R)


def compute_Z(R, g):
    """ Normalize R, assuming permutation distribution """
    nE     = g.number_of_edges()
    n      = g.number_of_nodes()
    deg_sq = (np.array(dict(g.degree).values()) ** 2).sum()
    
    tt     = np.arange(n) + 1
    mu_t   = nE * 2 * tt * (n - tt) / (n * (n - 1))
    p1_tt  = 2 * tt * (n - tt) / (n * (n - 1))
    p2_tt  = tt * (n - tt) * (n - 2) / (n * (n - 1) * (n - 2))
    p3_tt  = 4 * tt * (n - tt) * (tt - 1) * (n - tt - 1) / (n * (n - 1) * (n - 2) * (n - 3))
    A_tt   = (p1_tt - 2 * p2_tt + p3_tt) * nE + (p2_tt - p3_tt) * deg_sq + p3_tt * (nE ** 2)
    return (mu_t - R) / np.sqrt(A_tt - (mu_t ** 2) + 1e-7)

# --
# Run

if __name__ == "__main__":
    
    args = parse_args()
    
    # --
    # Form graph -- this part should be done externally probably...
    
    X      = pd.read_csv(args.inpath, header=None, sep='\t').values
    X_dist = squareform(pdist(X))
    mst    = minimum_spanning_tree(X_dist).tocoo()
    edges  = np.column_stack([mst.row, mst.col])
    g      = nx.from_edgelist(edges)
    
    # --
    # Run changepoint detection
    
    R = scan_stat(g)
    Z = compute_Z(R, g)
    
    # --
    # Write results
    
    df = pd.DataFrame({"R" : R, "Z" : Z})
    df.to_csv(args.outpath, sep='\t', index=False)
