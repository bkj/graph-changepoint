#### graph-changepoint

Implementation of [Graph-Based Change Point Detection](https://arxiv.org/abs/1209.1625.pdf)

#### Installation
```
conda create -n cp_env python=2.7 pip -y
source activate cp_env
pip install -r requirements.txt
```

#### Notes

ATM, we don't really care about the complexity of the distance computation or the similarity graph construction (MST).  We could construct the similarity graph lots of different ways -- minimum spanning tree, nearest neighbors, etc.