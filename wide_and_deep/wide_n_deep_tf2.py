import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers

CSV_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "gender",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income_bracket"
]

# feature_name: vocabulary
string_categorical_cols = {
    'gender': ["Female", "Male"],
    'education': [
        "Bachelors", "HS-grad", "11th", "Masters", "9th",
        "Some-college", "Assoc-acdm", "Assoc-voc", "7th-8th",
        "Doctorate", "Prof-school", "5th-6th", "10th", "1st-4th",
        "Preschool", "12th"
    ],
    'marital_status': [
        "Married-civ-spouse", "Divorced", "Married-spouse-absent",
        "Never-married", "Separated", "Married-AF-spouse", "Widowed"
    ],
    'relationship': [
        "Husband", "Not-in-family", "Wife", "Own-child", "Unmarried",
        "Other-relative"
    ],
    "workclass": [
        "Self-emp-not-inc", "Private", "State-gov", "Federal-gov",
        "Local-gov", "?", "Self-emp-inc", "Without-pay", "Never-worked"
    ]
}

# feature_name: boundaries
numeric_categorical_cols = {
   'age': [18, 25, 30, 35, 40, 45, 50, 55, 60, 65],
   'education_num': [3, 6, 9, 10, 12, 16],
   'capital_gain': [1000, 5000, 10000],
    'capital_loss': [100, 1000, 5000, 10000],
    'hours_per_week': [10, 20, 40, 45, 50, 60, 70]
}

cols_to_drop = [col for col in CSV_COLUMNS
                if (col not in string_categorical_cols) and (col not in numeric_categorical_cols)]

def df_to_dataset(data_frame, shuffle=True, batch_size=32):
    label = data_frame.pop('target')
    df = {key: value.to_numpy()[:, tf.newaxis] for key, value in data_frame.items()}
    ds = tf.data.Dataset.from_tensor_slices((dict(df), label))  # small datasets only
    if shuffle:
        ds = ds.shuffle(buffer_size=len(data_frame))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)
    return ds

def get_numeric_category_layer(boundaries):
    discretizer = tf.keras.layers.Discretization(bin_boundaries=boundaries)
    encoder = tf.keras.layers.CategoryEncoding(num_tokens=len(boundaries) + 1)
    return lambda feature: encoder(discretizer(feature))


def get_string_category_layer(vocab):
    index = layers.StringLookup(vocabulary=vocab)
    encoder = layers.CategoryEncoding(num_tokens=index.vocabulary_size())
    return lambda feature: encoder(index(feature))


train_file = 'datasets/census_income/adult.data'
test_file = 'datasets/census_income/adult.test'
df_train = pd.read_csv(train_file, header=None, names=CSV_COLUMNS)

df_test = pd.read_csv(test_file, header=None, names=CSV_COLUMNS)
# test file has Nan data
df_test = df_test.dropna()
for col in numeric_categorical_cols:
    df_test[col] = df_test[col].astype('int64')  # convert float to int

df_train['target'] = df_train["income_bracket"].apply(lambda x: ">50K" in x).astype(int)
df_test['target'] = df_test["income_bracket"].apply(lambda x: ">50K" in x).astype(int)
# Drop unused features and income_bracket.
df_train = df_train.drop(columns=cols_to_drop)
df_test = df_test.drop(columns=cols_to_drop)

batch_size = 64
train_ds = df_to_dataset(df_train, batch_size=batch_size)
test_ds = df_to_dataset(df_test, shuffle=False, batch_size=batch_size)

inputs = {}  # symbolic tensor-like object
encoded_features = []
for feature_name, vocab in string_categorical_cols.items():
    string_col = tf.keras.Input(shape=(1,), name=feature_name, dtype='string')
    inputs[feature_name] = string_col
    encoding_layer = get_string_category_layer(vocab)
    encoded_features.append(encoding_layer(string_col))
for feature_name, boundaries in numeric_categorical_cols.items():
    num_col = tf.keras.Input(shape=(1,), name=feature_name, dtype='int32')
    inputs[feature_name] = num_col
    encoding_layer = get_numeric_category_layer(boundaries)
    encoded_features.append(encoding_layer(num_col))

def create_linear_model():
    x = layers.concatenate(encoded_features)
    output = layers.Dense(1)(x)
    model = tf.keras.Model(inputs, output)
    return model


def test_linear_model():
    """Linear model only performance. """
    linear_model = create_linear_model()
    linear_model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["accuracy"],
        run_eagerly=True)
    linear_model.fit(train_ds, epochs=10)
    result = linear_model.evaluate(test_ds, return_dict=True)
    print(result)
