#!/usr/bin/env python
"""
    main.py
    
    Implementation of "Graph-Based Change Point Detection"
        https://arxiv.org/abs/1209.1625.pdf
"""

from __future__ import print_function, division

import sys
import argparse
import numpy as np
import pandas as pd
import networkx as nx

# --
# Helpers

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', type=str)
    parser.add_argument('--outpath', type=str)
    parser.add_argument('--two-d', action="store_true")
    return parser.parse_args()


def scan_stat(g, offset=0):
    """ apply scan stat the computes size of cut from offset to n """
    R, cross = [], []
    for n in g.nodes:
        for e in g.edges(n):
            if (e[0] >= offset) and ((e[1] > n) or (e[1] < offset)):
                cross.append(e)
        
        cross = [c for c in cross if (c[1] > n) or (c[1] < offset)]
        R.append(len(cross))
    
    return np.array(R)


def scan_stat_2d(g):
    """ apply scan stat for all possible offsets -- could be done in parallel """
    return np.vstack([scan_stat(g, offset=offset + 1) for offset in g.nodes])


def compute_Z(R, g):
    """ normalize 1d scan stats (by analytically computing mean and variance) """
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


def compute_Z_2d(R_2d, g):
    """ normalize 2d scan stats (by analytically computing mean and variance) """
    nE     = g.number_of_edges()
    n      = g.number_of_nodes()
    deg_sq = (np.array(dict(g.degree).values()) ** 2).sum()
    
    tt = np.arange(n)
    tt = np.vstack([tt - i for i in range(len(tt))])
    tt = tt.clip(min=0)
    
    mu_t  = nE * 2 * tt * (n - tt)/(n * (n - 1))
    p1_tt = 2 * tt * (n - tt)/(n * (n - 1))
    p2_tt = 4 * tt * (n - tt) * (tt - 1) * (n - tt - 1)/(n * (n - 1) * (n - 2) * (n - 3))
    V_tt  = p2_tt * nE + (p1_tt / 2 - p2_tt) * deg_sq + (p2_tt - (p1_tt ** 2)) * (nE ** 2)
    return (mu_t - R_2d) / np.sqrt(V_tt + 1e-7)


# def random_stat(g, num_nodes):
#     """ compute size of random cut """
#     nodes = set(np.random.choice(g.nodes, num_nodes))
#     return len([[e for e in g.edges(n) if e[1] not in nodes] for n in nodes])


def melt_array(x):
    """ helper for data formatting """
    df = pd.DataFrame(x)
    df['start'] = np.arange(df.shape[0])
    df = pd.melt(df, id_vars='start')
    df.columns = ('start', 'end', 'value')
    return df


# --
# Run

if __name__ == "__main__":
    
    args = parse_args()
    
    edges = pd.read_csv(args.inpath, header=None, sep='\t').values
    g     = nx.from_edgelist(edges)
    
    # --
    # Run changepoint detection
    
    if not args.two_d:
        print('main.py: computing scan stat', file=sys.stderr)
        R = scan_stat(g)
        
        print('main.py: normalizing', file=sys.stderr)
        Z = compute_Z(R, g)
        
        print('main.py: saving', file=sys.stderr)
        out_df = pd.DataFrame({"R" : R, "Z" : Z})
        
    else:
        print('main.py: computing scan stat', file=sys.stderr)
        R = scan_stat_2d(g)
        
        print('main.py: normalizing', file=sys.stderr)
        Z = compute_Z_2d(R, g)
        
        print('main.py: saving', file=sys.stderr)
        out_df = melt_array(R)
        out_df = out_df[out_df.start < out_df.end]
        out_df.columns = ('start', 'end', 'R')
        out_df['Z'] = melt_array(Z)['value']
    
    out_df.to_csv(args.outpath, sep='\t', index=False)

