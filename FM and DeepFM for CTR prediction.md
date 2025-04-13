
Paper
* [Factorization Machines](https://www.ismll.uni-hildesheim.de/pub/pdfs/Rendle2010FM.pdf)
* [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/abs/1703.04247)

## Factorization Machine

Factorization Machines (FM) model pairwise feature interactions as inner product of latent vectors between features.


## DeepFM Model
DeepFM  consists of two components, with shared inputs and an embedding layer.

### FM component

The FM component is a factorization machine to learn a linear (order-1) features, and order-2 feature interactions as inner product 




----
Capture second-order (pairwise) feature interactions without manually crafting cross-features.

Generalizes to unseen feature pairs, fewer parameters than one-hot


$\mathbf{v}_i, \mathbf{v}_j \in \mathbb{R}^k$ are **latent embedding vectors** for features  $i$ and $j$.


To avoid computing all $\binom{n}{2}$ pairs explicitly, FM uses this trick:

\frac{1}{2} \left( \left( \sum_{i=1}^{n} \mathbf{v}_i x_i \right)^2 - \sum_{i=1}^{n} \mathbf{v}_i^2 x_i^2 \right)

This lets you compute the entire second-order term in **linear time** with respect to feature dimension $n$, making it scalable.


These embeddings are **shared** across both:
- The **FM component** (for interaction modeling),
- The **Deep component** (as input to MLP).