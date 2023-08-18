# This script contains some additional analysis for extra insights into the review data.

import os
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer

#region DATA IMPORT
# Load pickled dataframe.
current_dir = os.path.dirname(os.path.abspath(__name__))
pickle_dir = 'processed data/processed_df.pickle'
export_path = os.path.join(current_dir, pickle_dir)
with open(export_path, 'rb') as f:
    df = pickle.load(f)
#endregion

# Let's try and find key differences between 4* and 5* reviews.
# This might help us identify what the difference is between a "good" product and a "great" product.
# We'll use the frequencies of each n-gram and look for differences between the two subsets.
# We're using a limited vocabulary to save computation time here (1000 n-grams).
count_vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=1000)
count_matrix = count_vectorizer.fit_transform(df.loc[(df["overall"] == 4) | (df["overall"] == 5), "stemmedText"])

# Get n-gram counts in each subset.
# Note: The .A1 method is being used to convert sparse matrices into 1D arrays.
four_matx = count_vectorizer.transform(df.loc[df["overall"] == 4, 'stemmedText'])
fours_sum = four_matx.sum(axis=0).A1
five_matx = count_vectorizer.transform(df.loc[df["overall"] == 5, 'stemmedText'])
fives_sum = five_matx.sum(axis=0).A1

# Get the n-grams with the highest frequencies in both subsets
top_ngrams_fours = [ngram for _, ngram in sorted(zip(fours_sum, count_vectorizer.get_feature_names_out()), reverse=True)[:250]]
top_ngrams_fives = [ngram for _, ngram in sorted(zip(fives_sum, count_vectorizer.get_feature_names_out()), reverse=True)[:250]]

# Identify top n-grams unique to four-star and five-star reviews
unique_ngrams_fours = set(top_ngrams_fours) - set(top_ngrams_fives)
unique_ngrams_fives = set(top_ngrams_fives) - set(top_ngrams_fours)

# A quick inspection of the unique terms suggest that some key features that attract 5* reviews are:
# - Nostalgia
# - Religious/Worship music
# - Soundtracks
# This may be useful information in identifying audiences to market towards and key themes to targeting in marketing materials.