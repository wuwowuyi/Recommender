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
hash_cols = {'occupation', 'native_country'}
crossed_cols = {
    'education_occupation': ['education', 'occupation'],
    'native_country_occupation': ['native_country', 'occupation'],
    'age_education_occupation': ['age', 'education', 'occupation']
}
cross_func = lambda x: ''.join(map(str, x))
target_col = 'income_bracket'
target_filter = lambda x: ">50K" in x


def get_dataset(selection, batch_size=64):
    def load_data():
        train_file = 'datasets/census_income/adult.data'
        test_file = 'datasets/census_income/adult.test'
        df_train = pd.read_csv(train_file, header=None, names=CSV_COLUMNS)
        df_test = pd.read_csv(test_file, header=None, names=CSV_COLUMNS)

        # test file has Nan data
        df_test = df_test.dropna()  # use df.isnull().values.any() to test
        for col in numeric_categorical_cols:
            df_test[col] = df_test[col].astype('int64')  # convert float to int

        df_train['target'] = df_train[target_col].apply(target_filter).astype(int)
        df_test['target'] = df_test[target_col].apply(target_filter).astype(int)

        # add cross columns
        for cross_feature, col_list in crossed_cols.items():
            df_train[cross_feature] = df_train[col_list].apply(cross_func, axis=1)
            df_test[cross_feature] = df_test[col_list].apply(cross_func, axis=1)

        return df_train, df_test

    def df_to_dataset(data_frame, shuffle=True, batch_size=32):
        label = data_frame.pop('target')
        df = {key: value.to_numpy()[:, tf.newaxis] for key, value in data_frame.items()}
        ds = tf.data.Dataset.from_tensor_slices((dict(df), label))  # small datasets only
        if shuffle:
            ds = ds.shuffle(buffer_size=len(data_frame))
        ds = ds.batch(batch_size)
        ds = ds.prefetch(batch_size)
        return ds

    df_train, df_test = load_data()
    cols_to_drop = (set(CSV_COLUMNS) | crossed_cols.keys()) - selection
    df_train = df_train.drop(columns=cols_to_drop)
    df_test = df_test.drop(columns=cols_to_drop)
    train_ds = df_to_dataset(df_train, batch_size=batch_size)
    test_ds = df_to_dataset(df_test, shuffle=False, batch_size=batch_size)
    return train_ds, test_ds


def get_inputs(selection: set[str]):
    def get_numeric_category_layer(boundaries):
        discretizer = layers.Discretization(bin_boundaries=boundaries)
        encoder = layers.CategoryEncoding(num_tokens=len(boundaries) + 1)
        return lambda feature: encoder(discretizer(feature))

    def get_string_category_layer(vocab):
        index = layers.StringLookup(vocabulary=vocab)
        encoder = layers.CategoryEncoding(num_tokens=index.vocabulary_size())
        return lambda feature: encoder(index(feature))

    def get_hash_category_layer(num_bins=1000):
        hasher = layers.Hashing(num_bins=num_bins)
        encoder = layers.CategoryEncoding(num_tokens=num_bins)
        return lambda feature: encoder(hasher(feature))

    inputs = {}
    features = []
    predicate = lambda x: x in selection
    for feature_name in filter(predicate, string_categorical_cols):
        string_col = tf.keras.Input(shape=(1,), name=feature_name, dtype='string')
        encoded = get_string_category_layer(string_categorical_cols[feature_name])(string_col)
        inputs[feature_name] = string_col
        features.append(encoded)
    for feature_name in filter(predicate, numeric_categorical_cols):
        num_col = tf.keras.Input(shape=(1,), name=feature_name, dtype='int32')
        encoded = get_numeric_category_layer(numeric_categorical_cols[feature_name])(num_col)
        inputs[feature_name] = num_col
        features.append(encoded)
    for feature_name in filter(predicate, hash_cols | crossed_cols.keys()):
        string_col = tf.keras.Input(shape=(1,), name=feature_name, dtype='string')
        encoded = get_hash_category_layer()(string_col)
        inputs[feature_name] = string_col
        features.append(encoded)
    return inputs, features


def create_linear_model(inputs, features):
    x = layers.concatenate(features)
    output = layers.Dense(1)(x)
    model = tf.keras.Model(inputs, output)
    return model


def create_deep_model(inputs, features):
    x = layers.concatenate(features)
    x = layers.Dense(100, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(50, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(1)(x)
    model = tf.keras.Model(inputs, output)
    return model


def create_wide_n_deep_model(wide_inputs, deep_inputs, wide_features, deep_features):
    # wide
    x_wide = layers.concatenate(wide_features)
    output_wide = layers.Dense(1)(x_wide)
    # deep
    x = layers.concatenate(deep_features)
    x = layers.Dense(100, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(50, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output_deep = layers.Dense(1)(x)
    outputs = output_wide + output_deep
    model = tf.keras.Model({**wide_inputs, **deep_inputs}, outputs)
    return model


def test_model(mode: str = 'wide_n_deep'):
    deep_cols = set([*string_categorical_cols.keys(), *hash_cols, *numeric_categorical_cols.keys()])
    deep_cols.remove('marital_status')
    if mode == 'wide_n_deep':
        wide_cols = set(crossed_cols.keys())
        cols = wide_cols | deep_cols
        wide_inputs, wide_features = get_inputs(wide_cols)
        deep_inputs, deep_features = get_inputs(deep_cols)
        model = create_wide_n_deep_model(wide_inputs, deep_inputs, wide_features, deep_features)
    elif mode == 'wide':
        base_cols = set([*string_categorical_cols.keys(), *hash_cols, 'age'])
        cols = base_cols | crossed_cols.keys()
        inputs, features = get_inputs(cols)
        model = create_linear_model(inputs, features)
    else:  # 'deep'
        cols = deep_cols
        inputs, features = get_inputs(cols)
        model = create_deep_model(inputs, features)

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["accuracy"],
        run_eagerly=True)
    train_ds, test_ds = get_dataset(cols)
    model.fit(train_ds, epochs=10)
    result = model.evaluate(test_ds, return_dict=True)
    print(result)


test_model()
