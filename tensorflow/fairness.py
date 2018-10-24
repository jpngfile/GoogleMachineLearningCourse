import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import tempfile
import seaborn as sns
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve

from IPython.core.display import display, HTML
import base64
import hopsfacets as facets
# facets is not pip-installable: https://github.com/PAIR-code/facets/issues/8
# from hopsfacets.feature_statistics_generator import FeatureStatisticsGenerator
# This exercise can only be done in a jupyter notebook for now
print("Modules are imported")

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "gender",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income_bracket"
]

train_df = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    names=COLUMNS,
    sep=r"\s*,\s*",
    engine="python",
    na_values="?"
)

test_df = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
    names=COLUMNS,
    sep=r"\s*,\s*",
    skiprows=[0],
    engine="python",
    na_values="?"
)

# Drop rows with missing values
train_df = train_df.dropna(how="any", axis=0)
test_df = test_df.dropna(how="any", axis=0)

print("UCI Adult Census Income dataset loaded.")

def csv_to_pandas_input_fn(data, batch_size=100, num_epochs=1, shuffle=False):
    return tf.estimator.inputs.pandas_input_fn(
        x=data.drop('income_bracket', axis=1),
        y=data['income_bracket'].apply(lambda x: ">50K" in x).astype(int),
        batch_size=batch_size,
        num_epochs=num_epochs,
        shuffle=shuffle,
        num_threads=1
    )

print("csv_to_pandas_input_fn() defined")

# Since we don't know the full range of possible values with occupation and
# native_country, we'll use categorical_column_with_hash_bucket() to help map
# each feature string into an integer ID.
occupation = tf.feature_column.categorical_column_with_hash_bucket(
    "occupation",
    hash_bucket_size=1000
)
native_country = tf.feature_column.categorical_column_with_hash_bucket(
    "native_country",
    hash_bucket_size=1000
)

# For the remaining categorical features, since we know that the possible values
# are, we can be more explicit and use categorical_column_with_vocabulary_list()
gender = tf.feature_column.categorical_column_with_vocabulary_list(
    "gender",
    ["Female", "Male"]
)
race = tf.feature_column.categorical_column_with_vocabulary_list(
    "race",
    ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"]
)
education = tf.feature_column.categorical_column_with_vocabulary_list(
    "education",
    [
        "Bachelors", "HS-grad", "11th", "Masters", "9th",
        "Some-college", "Assoc-acdm", "Assoc-voc", "7th-8th",
        "Doctorate", "Prof-school", "5th-6th", "10th", "1st-4th",
        "Preschool", "12th"
    ]
)
martial_status = tf.feature_column.categorical_column_with_vocabulary_list(
    "martial_status",
    [
        "Married-civ-spouse", "Divorced", "Married-spouse-absent",
        "Never-married", "Separated", "Married-AF-spouse", "Widowed"
    ]
)
relationship = tf.feature_column.categorical_column_with_vocabulary_list(
    "relationship",
    [
        "Husband", "Not-in-family", "Wife", "Own-child", "Unmarried",
        "Other-relative"
    ]
)
workclass = tf.feature_column.categorical_column_with_vocabulary_list(
    "workclass",
    [
        "Self-emp-not-inc", "Private", "State-gov", "Federal-gov",
        "Local-gov", "?", "Self-emp-inc", "Without-pay", "Never-worked"
    ]
)

print("Categorical feature columns defined")

# For Numeric features, we can just call on feature_column.numeric_column()
# to use its raw value instead of having to create a map between value and ID.
age = tf.feature_column.numeric_column("age")
fnlwgt = tf.feature_column.numeric_column("fnlwgt")
eduation_num = tf.feature_column.numeric_column("eduation_num")
capital_gain = tf.feature_column.numeric_column("capital_gain")
capital_loss = tf.feature_column.numeric_column("capital_loss")
hours_per_week = tf.feature_column.numeric_column("hours_per_week")

print("Numeric feature columns defined")

# We bucketize age since it has a lot of variance, to prevent overfitting
age_buckets = tf.feature_column.bucketized_column(
    age,
    boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65]
)

# List of variables, with special handling for gender subgroup
variables = [native_country, education, occupation, workclass, relationship, age_buckets]
subgroup_variables = [gender]
feature_columns = variables + subgroup_variables

deep_columns = [
    tf.feature_column.indicator_column(workclass),
    tf.feature_column.indicator_column(education),
    tf.feature_column.indicator_column(age_buckets),
    tf.feature_column.indicator_column(gender),
    tf.feature_column.indicator_column(relationship),
    tf.feature_column.embedding_column(native_country, dimension=8),
    tf.feature_column.embedding_column(occupation, dimension=8),
]

print(deep_columns)
print("Deep columns created.")

HIDDEN_UNITS = [1024, 512]
LEARNING_RATE = 0.1
L1_REGULARIZATION_STRENGTH = 0.0001
L2_REGULARIZATION_STRENGTH = 0.0001

model_dir = tempfile.mkdtemp()
single_task_deep_model = tf.estimator.DNNClassifier(
    feature_columns=deep_columns,
    hidden_units=HIDDEN_UNITS,
    optimizer=tf.train.ProximalAdagradOptimizer(
        learning_rate=LEARNING_RATE,
        l1_regularization_strength=L1_REGULARIZATION_STRENGTH,
        l2_regularization_strength=L2_REGULARIZATION_STRENGTH
    ),
    model_dir=model_dir
)

print("Deep neural net model defined.")

STEPS = 1000

single_task_deep_model.train(
    input_fn=csv_to_pandas_input_fn(train_df, num_epochs=None, shuffle=True),
    steps=STEPS
)

print("Deep neural net model is done fitting")

results = single_task_deep_model.evaluate(
    input_fn=csv_to_pandas_input_fn(test_df, num_epochs=1, shuffle=False),
    steps=None
)

print("model directory = %s" % model_dir)
print("--- Results ----")
for key in sorted(results):
    print("%s: %s" % (key, results[key]))


def compute_eval_metrics(references, predictions):
    tn, fp, fn, tp = confusion_matrix(references, predictions).ravel()
    precision = tp / float(tp + fp)
    recall = tp / float(tp + fn)
    false_positive_rate = fp / float(fp + tn)
    false_omission_rate = fn / float(tn + fn)
    return precision, recall, false_positive_rate, false_omission_rate

print("Binary confusion matrix and evaluation metrics defined.")

def plot_confusion_matrix(confusion_matrix, class_names, figsize = (8, 6)):
    # We're taking our calculated binary confusion matrix that's already in the form
    # of an array and turning it into a Pandas DataFrame because it's a lot
    # easier to work with when visualizing a heat map in Seaborn
    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)
    fig=plt.figure(figsize=figsize)

    # Combine the instance (numerical value) with its description
    strings = np.asarray([
        ["True Positives", "False Negatives"],
        ["False Positives", "True Negatives"]
    ])

    labels = (np.asarray(
        ["{0:d}\n{1}".format(value, string) for string, value in zip(strings.flatten(), confusion_matrix.flatten())]
    )).reshape(2, 2)

    heatmap = sns.heatmap(df_cm, annot=labels, fmt="")
    heatmap.yaxis.set_ticklabels(
        heatmap.yaxis.get_ticklabels(),
        rotation=0,
        ha="right"
    )
    heatmap.xaxis.set_ticklabels(
        heatmap.xaxis.get_ticklabels(),
        rotation=45,
        ha="right"
    )
    plt.title("Confusion Matrix ({})".format(SUBGROUP))
    plt.ylabel("References")
    plt.xlabel("Predictions")
    return fig

print("Binary confusion matrix visualization defined")

CATEGORY = "gender"
SUBGROUP = "Male"

# Given defined subgroup, generate predictions and obtain its corresponding ground truth
predictions_dict = single_task_deep_model.predict(input_fn=csv_to_pandas_input_fn(
    test_df.loc[test_df[CATEGORY] == SUBGROUP],
    num_epochs=1,
    shuffle=False
))
predictions = []
for prediction_item, in zip(predictions_dict):
    predictions.append(prediction_item["class_ids"][0])

actuals = list(test_df.loc[test_df[CATEGORY] == SUBGROUP]['income_bracket'].apply(lambda x: ">50K" in x).astype(int))
classes = ['Over $50K', 'Less than $50K']

# To stay consistent, we have to flip the confusion
# matrix around on both axes because sklearn's confusion matrix module by
# default is rotated.
rotated_confusion_matrix = np.fliplr(confusion_matrix(actuals, predictions))
rotated_confusion_matrix = np.flipud(rotated_confusion_matrix)

plot_confusion_matrix(rotated_confusion_matrix, classes)
precision, recall, fpr, fomr = compute_eval_metrics(actuals, predictions)
print("Precision: %.4f" % precision)
print("Recall: %.4f" % recall)
print("False Positive Rate: %.4f" % fpr)
print("False omission Rate: %.4f" % fomr)
plt.show()
