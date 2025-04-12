Paper [Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/abs/1706.06978)

## Base Model

Embedding layer converts input one-hot or multi-hot sparse binary vectors to dense embeddings.

For one-hot input
```python
one_hot_input = tf.constant([
    [0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0]
], dtype=tf.int32)
indices = tf.argmax(one_hot_input, axis=-1)
embedding_layer = tf.keras.layers.Embedding(input_dim=5, output_dim=4)
```

For multi-hot input
```python
import tensorflow as tf

# Simulated input: batch of multi-hot binary vectors (dense format)
multi_hot_input = tf.constant([
    [0, 1, 0, 1, 0, 1],   # user 1 clicked items 1, 3, 5
    [1, 0, 0, 0, 1, 0]    # user 2 clicked items 0, 4
], dtype=tf.float32)
vocab_size = 6
embedding_dim = 4
# Get the active indices (nonzero elements) for each row
nonzero_indices = tf.where(tf.not_equal(multi_hot_input, 0))
# Gather item indices for each sample
item_ids = tf.RaggedTensor.from_value_rowids(
    values=nonzero_indices[:,1],  # the item IDs (column indices)
    value_rowids=nonzero_indices[:,0],  # which row each one belongs to
    nrows=tf.cast(tf.shape(multi_hot_input)[0], tf.int64)
)
# embedded_items is a 3-D raggedTensor (batch_size, number_of_non_zeros, embedding_dim)
# Pooling converts it to (batch_size, embedding_dim)
embedded_items = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)(item_ids)
# NOTE: paper use sum pooling, I use average pool here simply 
# because tensorflow does not have a sum pooling layer, and for simplicity just use average pooling.
dense_vector = tf.keras.layers.GlobalAveragePooling1D()(embedded_items)
```

### Loss
Loss is simply logistic regression predicting whether an Ad is clicked or not.

$\displaystyle L = -\frac{1}{N}\sum_i\left[y_i\log p(x_i) + (1-y_i)\log (1-p(x_i))\right]$.


## Local Activation Unit (Attention)

DIN calculate the representation vector of user interests by taking into account the relevance of historical behavior w.r.t. candidate ad. 

🤔 [Deep Neural Networks for YouTube Recommendations](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45530.pdf) talked about similar idea:
> the most important signals are those that describe a user's previous interactions with the item itself and other similar items.

DIN introduced a local activate unit (Attention) calculate user's interests in the candidate ad.

$v_U(A) = f(v_A, e_1, e_2, ..., e_H) = \sum_{j=1}^H a(e_j, v_A)e_j = \sum_{j=1}^H w_je_j$.

where
* $\{e_1, e_2, ..., e_H\}$ is the click history of user $U$
* candidate ad $v_A$
* $a(\cdot)$ is a feed-forward network with output as the activation weight $w_j$ for each history $e_j$

Different from attention methods, the constraint $\sum_iW_i$ is relaxed. This value is treated as **an approximation of the intensity of activated user interests to some degree**. 

The authors also tried LSTM to model user history, but didn't get improvement.

Different from text which is under the constraint of grammar, user history sequence may contain multiple concurrent interests. **Rapid jumping and sudden ending over these interests causes the sequence data of user behaviors to seem to be noisy**. A possible direction is to design special structure to model such data in a sequence way.

See [code here](https://github.com/zhougr1993/DeepInterestNetwork/blob/9765f96202f849e59ff260c8b46931a0ddf01d77/din/model.py#L200) to understand how this attention layer works.



