# Drop low dates from dataframe

import pandas as pd
import numpy as np
from rsub import *
from matplotlib import pyplot as plt

df = pd.read_csv('./yankee.tsv', header=None, sep='\t')
sub = df[df[2] > np.percentile(df[2], 10)]
sub = sub.sort_values(2).reset_index(drop=True)
sub[[0, 1]].to_csv('./yankee-filtered.tsv', header=None, sep='\t', index=False)