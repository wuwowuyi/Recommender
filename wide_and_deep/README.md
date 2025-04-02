Paper [Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792). 

**memorization (wide)** + **generalization (deep)**

Wide branch memorize patterns like:  
* `"action"` → likely to click "Mad Max"  
* `"romantic"` → likely to click "The Notebook"

Deep branch can generalize (e.g., "loves" ≈ "enjoys").


`wide_n_deep_tutorial.py` is the original tutorial on the paper. It does not run, and is kept here only for a reference.

`wide_n_deep_tf2.py` is my rewrite in Tensorflow (2.1+).
Note for simplicity, the preprocessing here does not follow the best practice. See [My summary on the best practice processing structured data](../Structured%20Data%20Preprocessing%20Best%20Practice.md)

