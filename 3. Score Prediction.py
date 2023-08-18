# This script contains the classification modelling, using the review text to estimate the review score.

import numpy as np
import os
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils.class_weight import compute_class_weight

# Parse the report into a DataFrame
def csv_report(report, name):
    lines = report.split('\n')
    data = [line.split()[1:] for line in lines[2:-5]]
    columns = ['class', 'precision', 'recall', 'f1-score', 'support']
    df = pd.DataFrame(data, columns=columns)

    output_csv_path = name+".csv"
    df.to_csv(output_csv_path)

#region DATA IMPORT
# Load pickled dataframe.
current_dir = os.path.dirname(os.path.abspath(__name__))
pickle_dir = 'processed data/processed_df.pickle'
export_path = os.path.join(current_dir, pickle_dir)
with open(export_path, 'rb') as f:
    df = pickle.load(f)
#endregion

# Prep features.
vectorizer = CountVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(df['stemmedText'])
y = df["overall"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#region RANDOM FOREST MODEL
# Initialise and run model using a grid search to find parameters, and evaluating using f1-score with cross-validation.
# We could do a much more comprehensive parameter search, but due to computation time on my machine, we'll keep it small.
rf = RandomForestClassifier(random_state=42)
rf_params = {'max_depth': [None, 10],
             'min_samples_split': [10, 20],
             'min_samples_leaf': [4, 8]}

rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring="f1_macro")
rf_grid.fit(X_train, y_train)
rf_best_params = rf_grid.best_params_

best_rf = RandomForestClassifier(**rf_best_params, random_state=42)
best_rf.fit(X_train, y_train)
rf_pred = best_rf.predict(X_test)
rf_report = classification_report(y_test, rf_pred)
csv_report(rf_report, "rf")
# Let's try class weighting due to imbalanced classes.
# This is less computationally expensive than oversampling, and the dataset is already large (and slow).
# Ideally we'd re-do a grid search, but for the sake of computation time we'll just re-use our parameters from earlier.
class_weights = compute_class_weight('balanced', classes=y_train.unique(), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

rf_weighted = RandomForestClassifier(**rf_best_params, class_weight=class_weight_dict, random_state=42)
rf_weighted.fit(X_train, y_train)
rf_wighted_pred = rf_weighted.predict(X_test)
rf_weighted_report = classification_report(y_test, rf_wighted_pred)
csv_report(rf_weighted_report, "rf_weighted")
#endregion

#region NAIVE BAYES CLASSIFIER
# Initialise and run model using a grid search to find parameters, and evaluating using f1-score with cross-validation.
nb = MultinomialNB()
nb_params = {'alpha': [0.1, 1, 10],
             'fit_prior': [True, False]}

nb_grid = GridSearchCV(nb, nb_params, cv=5, scoring="f1_macro")
nb_grid.fit(X_train, y_train)
nb_best_params = nb_grid.best_params_

best_nb = MultinomialNB(**rf_best_params)
best_nb.fit(X_train, y_train)
nb_pred = best_nb.predict(X_test)
nb_report = classification_report(y_test, nb_pred)
csv_report(nb_report, "nb")

# Let's try class weighting here too.
# Manually set class prior probabilities based on class weights
nb_weighted = MultinomialNB(**nb_best_params, class_prior=class_weights)
nb_weighted.fit(X_train, y_train)
nb_weighted_pred = nb_weighted.predict(X_test)
nb_weighted_report = classification_report(y_test, nb_weighted_pred)
csv_report(nb_weighted_report, "nb_weighted")
#endregion

#region MODEL EVALUATION
# Now we can compare model performance.
#endregion


