Best practice in preprocessing numerical, categorical, timestampe/date, short text for recommenders.

Use [Keras preprocessing layers](https://keras.io/api/layers/preprocessing_layers/) for example code.
Suppose we have a `DataFrame` `df` that contains the data. 

In general
* DNNs prefer dense inputs
* Tree models like XGBoost uses same encoding strategies as linear models


## Numerical data

### For Linear models
For linear models, when the relationship is **not** approximately linear between feature and target, like age or income, discretize and bucketing.

```python
age_bound = [18, 25, 30, 35, 40, 45, 50, 55, 60, 65]
col = tf.keras.Input(shape=(1,), name=feature_name, dtype='int32')
discretizer = Discretization(bin_boundaries=age_bound)
encoder = CategoryEncoding(num_tokens=len(age_bound)+1, output_mode='one_hot')
encoded_age = encoder(discretizer(col)

# or equivalently
discretizer = Discretization(bin_bound=age_bound, output_mode="one_hot")
encoded_age = discretizer(input)
```

When the relationship is **approximately linear** between feature and target, normalize it like the input to DNNs. 

### For Deep Models
For deep models, and **for a wide (linear) model when we are confident the relationship between that feature and the target is linear or approximately linear**, `Normalize` the data.

```python
norm_layer = layers.Normalization()
norm_layer.adapt(df["some_number"].values[:, None])
num_input = tf.keras.Input(shape=(1,), dtype=tf.float32, name="some_number")
normalized_num = norm_layer(age_input)
```

#### Why Don't We Typically Discretize for Deep Models?
(From ChatGPT)

 ✅ 1. **Deep models can learn nonlinear patterns directly**
- DNNs can **approximate arbitrary functions**.
- If age has a nonlinear relationship with your target (e.g., clicks peak at age 25–35), a deep model can **learn that pattern** from raw continuous input.
> No need to hand-engineer bins — deep models learn their own "bins" internally via learned weights.

 ✅ 2. **Discretization destroys granularity**
- If you bucket age 25 and 39 into the same bin `[20–40)`, the model **can’t tell them apart**, even though they’re very different.
- Continuous inputs preserve more **fine-grained information**, which DNNs thrive on.

 ✅ 3. **Discretization = manual inductive bias**
- It imposes **hard thresholds** based on assumptions.
- Might work well if your bins are meaningful (e.g., "teen", "adult", "senior"), but if not carefully tuned, it can **introduce error** or degrade performance.
- 
 ✅ 4. **Gradient flow is better with continuous features**
- DNNs rely on **smooth gradients** to optimize.
- Discretization introduces **sudden jumps (non-differentiable boundaries)**, which can make learning harder or unstable — especially if you apply binning too early in the pipeline.

## Categorical (string or int)

In general Categorical features model **group-wise effects**.
### For Linear models
* Use `StringLookup` or `IntegerLookup` if vocab is stable and small. e.g., country, gender
* Use `Hashing` for unknown vocab or massive cardinality. e.g., user_id, tags

Use `one_hot` for single categorical columns where each sample has exactly one category like gender, userid. 
Use `multi_hot` for multi-categorical where each sample may have multiple categories like tags.
A multi-hot vector indicates that multiple categories (or features) are active for a given input sample.

### For Deep models
If the categorical feature has a **stable and small vocabulary**
* Use `StringLookup` or `IntegerLookup` to turn strings/ints into indices  
* Use `Embedding` to map those indices to dense, trainable vectors 
* Flatten before feeding into DNNs

For a categorical feature is `"user_tier"` with values like `["free", "silver", "gold", "platinum"]`
```python
vocab = ["free", "silver", "gold", "platinum"]
user_tier_input = tf.keras.Input(shape=(1,), dtype=tf.string, name="user_tier")
lookup = layers.StringLookup(vocabulary=vocab, output_mode="int")
embedding = layers.Embedding(input_dim=len(vocab) + 1, output_dim=4)
tier_index = embedding(lookup(user_tier_input))  # shape: (batch, 1, 4)
# Flatten or Reshape before feeding into dense
tier_vector = layers.Flatten()(embedding)  # shape: (batch, 4)
```

For High-Cardinality Categorical Features, like user_id, movie_id, product_sku, etc.
```python
user_input = tf.keras.Input(shape=(1,), dtype=tf.string, name="user_id")
num_bins = 1_000_000 # 1M bins
hasher = layers.Hashing(num_bins=num_bins, output_mode='int')  
encoder = layers.Embedding(input_dim=num_bins, output_dim=32)
userid_embedding = encoder(hasher(user_input))
user_vector = layers.Flatten()(user_embedding)
```

Embedding size depends on model size, cardinality, data volume etc. For small cardinality, the common values are 4, 8, etc. For high cardinality, common values are 32, 64, etc. 

## Timestamp, Dates

Typically timestamps and Dates are decomposed into year, month, weekday, hour etc. before preprocessing. 
If we use a Date like `Year-month-day` directly, treat it like a Categorical property with high cardinality.

| Feature          | Usefulness                                                                                                                                                                                            | Type                   |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------- |
| Hour of day      | Daily cycle patterns (sleep/work)                                                                                                                                                                     | Cyclic (0–23)          |
| Day of week      | Weekday/weekend behavior (Cyclic)<br>Some per weekday pattern (Categorical)                                                                                                                           | Cyclic or Categorical  |
| Month            | Seasonal patterns (Cyclic)<br>Month specific, like school semester, holiday behavior (Categorical)                                                                                                    | Cyclic or Categorical  |
| Year             | Time since an event, like time since creation to indicate whether it is new or old (Numeric)<br>Pattern or distribution shift (group-wise, use Categorical)<br>For cross features, just use embedding | Numeric or Categorical |
| Is weekend       | Binary feature                                                                                                                                                                                        | Boolean                |
| Time since event | Time since user last interacted, <br>Time since item was item/session/user created                                                                                                                    | Numeric                |
| Unix timestamp   | Useful for sorting, sometimes embedding                                                                                                                                                               | Numeric                |

Example of Cyclic encoding

```Python
def cyclic_encoding(x, max_value):
    x = tf.cast(x, tf.float32)
    sin = tf.math.sin(2 * np.pi * x / max_value)
    cos = tf.math.cos(2 * np.pi * x / max_value)
    return tf.stack([sin, cos], axis=-1)
    
df["timestamp"] = pd.to_datetime(df["timestamp"], unit='s')
df["hour"] = df["timestamp"].dt.hour
df["weekday"] = df["timestamp"].dt.weekday

hour = tf.keras.Input(shape=(1,), name="hour")
weekday = tf.keras.Input(shape=(1,), name="weekday")
hour_encoded = layers.Lambda(lambda x: cyclic_encoding(x, 24))(hour)
weekday_encoded = layers.Lambda(lambda x: cyclic_encoding(x, 7))(weekday)
# ... remember to Flatten before concatenation!
```

Use **Cyclic Encoding** When:
- There is a **continuous or periodic** relationship
- the "**distance**" between values matters. Like Monday and Tuesday are closer than Monday and Wednesday.
- To preserve continuity for DNNs
Example: hour=23 is close to hour=0 (same for weekday=6 and weekday=0)

Use **Categorical Encoding** When:
- The values are better treated as **separate groups with no implied order**, for example, no assumption of a smooth relationship between Monday and Tuesday
- Use embeddings for DNNs (e.g., learn behavior on each day/month independently)
- Use them in feature crosses

If we want both, use both and concatenate.

**NOTE**
- Do not using future timestamps when predicting current labels
- Time-based features are **causal** (e.g., don’t use “days until click”)

## Boolean

For linear models, we can treat a boolean column like a Categorical column whose vocabulary size is 2. 
```python
is_active = tf.keras.Input(shape=(1,), dtype=tf.bool, name="is_active")
indexer = tf.keras.layers.StringLookup(vocabulary=["False", "True"])
encoder = tf.keras.layers.CategoryEncoding(num_tokens=2, output_mode="one_hot")
active_feature = encoder(indexer(is_active))
```

For deep models, cast to `float32`. 
```python
active = tf.keras.Input(shape=(1,), dtype=tf.bool, name="is_active")
active_feature = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32))(active)
```

### Multi-hot Sparse Input
When there are many boolean columns, where "many" means hundreds or more, and most values are zero, we can use multi-hot sparse input.
Example use case: movie viewing history, click history, tags, text.

See [A working example](#a-full-working-example) below. In the example, the column `clicked_genres` is a multi-hot sparse input. Remember to use a `Dense` layer to convert it to a dense multi-hot vector before feeding into DNNs.

## Text Features

short text data like user profile, brief description
* For linear models,  typically `TextVectorization(output_mode='tf_idf')`
* For deep models, `TextVectorization` → `Embedding`

For deep and wide models, TF-IDF scores complements the learned embeddings in the deep, combining **memorization (wide)** + **generalization (deep)**
Wide branch memorize patterns like:  
* `"action"` → likely to click "Mad Max"  
* `"romantic"` → likely to click "The Notebook"
Deep branch can generalize (e.g., "loves" ≈ "enjoys").

```python
user_profile = tf.keras.Input(shape=(1,), dtype=tf.string, name="user_profile")

# wide
wide_vectorizer = layers.TextVectorization(max_tokens=1000, output_mode="tf_idf")
wide_vectorizer.adapt(df["user_profile"])
wide_text = wide_vectorizer(user_profile)  # shape: (batch, vocab_size)

# deep
deep_vectorizer = layers.TextVectorization(  
    max_tokens=1000,  
    output_mode="int",  
    output_sequence_length=10  
)  
deep_vectorizer.adapt(df["user_profile"])  
deep_text = deep_vectorizer(inputs["user_profile"])
# embedding shape is (batch_size, sequence_length, embedding_dim)
deep_text = layers.Embedding(input_dim=1000, output_dim=16)(deep_text)
# average pooling before feeding into a Dense layer
deep_text = layers.GlobalAveragePooling1D()(deep_text)  # (batch_size, 16)
```
The average pooling acts like a semantic **bag of words** — capturing the overall meaning of the text. 

Note that the `TextVectorization` layer can only be executed on a CPU, so put the `TextVectorization` layer in the `tf.data`pipeline.

## A full working example

```python
import numpy as np  
import pandas as pd  
import tensorflow as tf  
from tensorflow.keras import layers  
  
df = pd.DataFrame({  
    "clicked_genres": [[0, 2, 5], [1, 4], [3], [0, 1, 5]],  # sparse input  
    "age": [25, 30, 22, 40],  # numeric  
    "is_premium": [True, False, True, False],  # boolean  
    "user_type": ["free", "premium", "guest", "vip"],  # categorical  
    "timestamp": [1490195805, 1490191000, 1490205805, 1490235805],  # time  
    "user_profile": [  # text  
        "loves action and adventure movies",  
        "interested in sci-fi and tech",  
        "enjoys drama and romantic films",  
        "likes horror and thrillers"  
    ],  
    "label": [1, 0, 1, 0]  
})  
df["timestamp"] = pd.to_datetime(df["timestamp"], unit='s')  
df["hour"] = df["timestamp"].dt.hour  
df["weekday"] = df["timestamp"].dt.weekday  
df.pop("timestamp")  
  
# vocabs  
genre_vocab = ["action", "comedy", "drama", "horror", "romance", "sci-fi"]  
user_type_vocab = ["free", "premium", "guest"]  
  
  
def cyclic_encoding(x, max_value):  
    """For hour and weekday to capture a periodic pattern. """  
    x = tf.cast(x, tf.float32)  
    sin = tf.math.sin(2 * np.pi * x / max_value)  
    cos = tf.math.cos(2 * np.pi * x / max_value)  
    return tf.stack([sin, cos], axis=-1)  
  
  
def df_to_dataset(dataframe, batch_size=2, shuffle=False):  
    labels = dataframe.pop("label").values.astype("float32")  
  
    # For the column 'clicked_genres' (a column of lists), create a RaggedTensor.  
    # (Other columns can be converted to tensors directly.)    clicked_genres = dataframe.pop("clicked_genres")  
    features = {col: tf.convert_to_tensor(dataframe[col].values) for col in dataframe.columns}  
  
    # Add the clicked_genres as a RaggedTensor.  
    features["clicked_genres"] = tf.ragged.constant(clicked_genres.values, dtype=tf.int32)  
  
    ds = tf.data.Dataset.from_tensor_slices((features, labels))  
    if shuffle:  
        ds = ds.shuffle(buffer_size=len(dataframe))  
    # asynchronously on CPU, and will be buffered before going into the model  
    ds = ds.batch(batch_size)  
    ds = ds.prefetch(batch_size)  
    return ds  
  
  
raw_ds = df_to_dataset(df.copy(), batch_size=2)  
  
  
def preprocess(features, label):  
    # For clicked_genres, features["clicked_genres"] is a RaggedTensor of shape [batch, None].  
    # For each sample, we convert the list of indices into a multi-hot vector.    def multi_hot_fn(genre_indices):  
        # genre_indices: 1D tensor of genre indices for one sample.  
        one_hot = tf.one_hot(genre_indices, depth=len(genre_vocab))  # shape: [batch_size, genre_vocab_size]  
        multi_hot = tf.reduce_sum(one_hot, axis=0)  # shape: [genre_vocab_size]  
        return multi_hot  
  
    # Replace the ragged column with the dense multi-hot vector.  
    features["clicked_genres"] = tf.map_fn(multi_hot_fn, features["clicked_genres"], fn_output_signature=tf.float32)  
    return features, label  
  
  
# Map the preprocessing function to the dataset.  
processed_ds = raw_ds.map(preprocess)  
  
# Define input layers matching the keys in our dataset.  
inputs = {  
    "clicked_genres": tf.keras.Input(shape=(len(genre_vocab),), sparse=True, name="clicked_genres"),  
    "age": tf.keras.Input(shape=(1,), dtype=tf.float32, name="age"),  
    "is_premium": tf.keras.Input(shape=(1,), dtype=tf.bool, name="is_premium"),  
    "user_type": tf.keras.Input(shape=(1,), dtype=tf.string, name="user_type"),  
    "hour": tf.keras.Input(shape=(1,), name="hour"),  
    "weekday": tf.keras.Input(shape=(1,), name="weekday"),  
    "user_profile": tf.keras.Input(shape=(1,), dtype=tf.string, name="user_profile")  
}  
  
norm_age = layers.Normalization()  
norm_age.adapt(df['age'].values[:, None])  # must be 2D because Keras assume that the data is batched  
age_encoded = norm_age(inputs["age"])  
  
# Boolean: Cast to float via a Lambda layer  
premium_float = layers.Lambda(lambda x: tf.cast(x, tf.float32))(inputs["is_premium"])  
  
# Categorical: StringLookup + Embedding for user_type  
user_lookup = layers.StringLookup(vocabulary=user_type_vocab)  
user_id = user_lookup(inputs["user_type"])  
user_emb = layers.Embedding(input_dim=len(user_type_vocab) + 1, output_dim=4)(user_id)  
# Flatten the embeddings (from shape [batch, 1, 4] to [batch, 4])  
user_emb = layers.Flatten()(user_emb)  
  
# For clicked_genres, we already have a dense multi-hot vector.  
# Pass it through a Dense layer to get a representation.  
genre_dense = layers.Dense(16, activation="relu")(inputs["clicked_genres"])  
  
# time columns  
hour_encoded = layers.Lambda(lambda x: cyclic_encoding(x, 24))(inputs["hour"])  
weekday_encoded = layers.Lambda(lambda x: cyclic_encoding(x, 7))(inputs["weekday"])  
hour_encoded = layers.Flatten()(hour_encoded)  
weekday_encoded = layers.Flatten()(weekday_encoded)  
  
# user profile  
wide_vectorizer = layers.TextVectorization(max_tokens=1000, output_mode="tf_idf")  
wide_vectorizer.adapt(df["user_profile"])  
wide_text = wide_vectorizer(inputs["user_profile"])  # shape: (batch, vocab_size)  
  
deep_vectorizer = layers.TextVectorization(  
    max_tokens=1000,  
    output_mode="int",  
    output_sequence_length=10  
)  
deep_vectorizer.adapt(df["user_profile"])  
embedding_dim = 16  
deep_text = deep_vectorizer(inputs["user_profile"])  
deep_text = layers.Embedding(input_dim=1000, output_dim=embedding_dim)(deep_text)  # (batch_size, sequence_length, embedding_dim)  
# acts like a semantic bag of words — capturing the overall meaning of the text.  
deep_text = layers.GlobalAveragePooling1D()(deep_text)  # (batch_size, embedding_dim)  
  
  
# Concatenate all features for deep models  
all_features = layers.Concatenate()([  
    genre_dense,  
    age_encoded,  
    premium_float,  
    user_emb,  
    hour_encoded,  
    weekday_encoded,  
    deep_text  
])  
# Build a simple DNN  
deep_x = layers.Dense(32, activation="relu")(all_features)  
deep_x = layers.Dense(16, activation="relu")(deep_x)  
combined = layers.Concatenate()([deep_x, wide_text])  
output = layers.Dense(1, activation="sigmoid")(combined)  
  
model = tf.keras.Model(inputs=inputs, outputs=output)  
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])  
model.summary()  
model.fit(processed_ds, epochs=5)

```