# This script trials a couple of different modelling approaches using Random Forest and Naive Bayes classifiers
# to try and predict review scores from the review text.

import numpy as np
import os
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils.class_weight import compute_class_weight

# This function will be used to save a number of model metrics to .csv files for evaluation after each model has been run.
def csv_report(report, name):
    lines = report.split("\n")
    data = [line.split()[1:] for line in lines[2:-5]]
    columns = ["precision", "recall", "f1-score", "support"]
    df = pd.DataFrame(data, columns=columns)

    output_csv_path = name+".csv"
    df.to_csv(output_csv_path)

#region DATA IMPORT
# Load pickled dataframe.
current_dir = os.path.dirname(os.path.abspath(__name__))
pickle_dir = "processed data/processed_df.pickle"
export_path = os.path.join(current_dir, pickle_dir)
with open(export_path, "rb") as f:
    df = pickle.load(f)
#endregion

# Prep features.
# Here we transform our review text into an n-gram count matrix (considering only unigrams and bigrams).
# We use a stratified split due to the severely unbalanced classes we saw in our EDA.
vectorizer = CountVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(df["stemmedText"])
y = df["overall"]
stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(stratified_split.split(X, y))
X_train = X[train_idx]
X_test = X[test_idx]
y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

#region RANDOM FOREST MODELS
# Train a number of classification models, and output evaluation metrics using the function we defined above.
# We will try using the dataset as-is, as well as using class weights to improve performance on highly unbalanced classes.
# In all cases we will use f1-score as our primary metric, again because we want to assess performance on unbalanced classes.

# The first model we trial is a Random Forest classifier, using a grid search to tune hyperparameters.
# The GridSearchCV funtion builds in k-fold cross validation for more reliable evaluation.
# We could do a much more comprehensive parameter search, but due to computation time we'll keep it small.
rf = RandomForestClassifier(random_state=42)
rf_params = {"max_depth": [None, 10],
             "min_samples_split": [10, 20],
             "min_samples_leaf": [4, 8]}

rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring="f1_macro")
rf_grid.fit(X_train, y_train)
rf_best_params = rf_grid.best_params_

best_rf = RandomForestClassifier(**rf_best_params, random_state=42)
best_rf.fit(X_train, y_train)
rf_pred = best_rf.predict(X_test)
rf_report = classification_report(y_test, rf_pred)
csv_report(rf_report, "rf")

# We will re-use the hyperparameters chosen above to train another Random Forest classifier using class weights for comparison.
# Ideally we would re-tune hyperparameters, but this is simpler for demonstration purposes.
# An alternative approach could be oversampling, but class weighting is less computationally expensive and run time is a major concern here.
class_weights = compute_class_weight("balanced", classes=y_train.unique(), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

rf_weighted = RandomForestClassifier(**rf_best_params, class_weight=class_weight_dict, random_state=42)
rf_weighted.fit(X_train, y_train)
rf_wighted_pred = rf_weighted.predict(X_test)
rf_weighted_report = classification_report(y_test, rf_wighted_pred)
csv_report(rf_weighted_report, "rf_weighted")
#endregion

#region NAIVE BAYES CLASSIFIERS
# We repeat the above steps using a Multinomial Naive Bayes classifier.
# First we train a model using a grid search for hyperparameter tuning.
# then we re-use the best performing parameters to train a second NB classifier using class weights.
nb = MultinomialNB()
nb_params = {"alpha": [0.1, 1, 10],
             "fit_prior": [True, False]}

nb_grid = GridSearchCV(nb, nb_params, cv=5, scoring="f1_macro")
nb_grid.fit(X_train, y_train)
nb_best_params = nb_grid.best_params_

best_nb = MultinomialNB(**rf_best_params)
best_nb.fit(X_train, y_train)
nb_pred = best_nb.predict(X_test)
nb_report = classification_report(y_test, nb_pred)
csv_report(nb_report, "nb")

# Class weighted NB model.
# Manually set class prior probabilities as the class weights
nb_weighted = MultinomialNB(**nb_best_params, class_prior=class_weights)
nb_weighted.fit(X_train, y_train)
nb_weighted_pred = nb_weighted.predict(X_test)
nb_weighted_report = classification_report(y_test, nb_weighted_pred)
csv_report(nb_weighted_report, "nb_weighted")
#endregion

#region MODEL EVALUATION
# Now we can compare model performance from the output .csvs.
# NOTE: These .csvs were produced through the GCP notebooks using only a subset of the full dataset.
#       See "/gcp notebooks" for full code, although model code is identical.
rf_report = pd.read_csv("gcp notebooks/rf.csv")
rf_weighted_report = pd.read_csv("gcp notebooks/rf_weighted.csv")
nb_report = pd.read_csv("gcp notebooks/nb.csv")
nb_weighted_report = pd.read_csv("gcp notebooks/nb_weighted.csv")

# Let's look at macro and weighted-average f1-scores.
def weighted_f1(df, name):
    df["weighted_f1"] = df["f1-score"] * df["support"]
    weighted_average_f1 = df["weighted_f1"].sum() / df["support"].sum()
    print(name, ": ", weighted_average_f1)

def macro_f1(df, name):
    macro_f1 = df["f1-score"].mean()
    print(name, ": ", macro_f1)

# Weighted averages
weighted_f1(rf_report, "rf")
weighted_f1(rf_weighted_report, "rf_weighted")
weighted_f1(nb_report, "nb")
weighted_f1(nb_weighted_report, "nb_weighted")

# Macro scores
macro_f1(rf_report, "rf")
macro_f1(rf_weighted_report, "rf_weighted")
macro_f1(nb_report, "nb")
macro_f1(nb_weighted_report, "nb_weighted")


# With the best weighted- and macro-f1 averages, we could say our "overall" best performer is the unweighted Naive Bayes classifier.

# However, this depends on our use for the model. E.g. if we wanted to specifically focus on identifying the differences between
# 4* and 5* reviews (see Additional Insights script!), we might want to look at the weighted Random Forest model which performs
# better in those specific two classes.

#endregion


