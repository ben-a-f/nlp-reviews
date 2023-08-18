# This script cleans and preps the data for analysis.

# Dataset: Amazon Reviews - Digital Music
# Source: https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/
# NOTE: The dataset is not included in this Git repository at present, as even when compressed it is too large. You will have to download it yourself from the link above
#       if you want to rerun any of this analysis.
#       I am looking into Git LFS to get around this restriction and will hopefully resolve this shortly.

import gzip
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import os
import pandas as pd
import pickle
import string

#region DATA IMPORT
# Get data filepath.
current_dir = os.path.dirname(os.path.abspath(__name__))
data_dir = 'raw data/Digital_Music.json.gz'
import_path = os.path.join(current_dir, data_dir)

# Read zipped JSON file into dataframe.
with gzip.open(import_path, 'rb') as f:
    df = pd.read_json(f, lines=True)
#endregion

#region DATA CLEANING
# Drop unnecessary columns for this project.
df = df.drop(["reviewTime", "reviewerID", "asin", "reviewerName", "summary", "unixReviewTime", "vote", "image"], axis=1)

# Check NAs and column types.
df.info()

# Drop rows with no review text.
df = df.dropna(subset=["reviewText"])

# Check all scores are values 1-5. We convert to categorical because the scores aren't continuous.
df["overall"].unique()
df["overall"] = df['overall'].astype('category')

# Check style values.
# We clean up the formatting with regex. This also conveniently converts "nan" strings into proper nans.
df["style"] = df["style"].astype(str)
# Discard the 'format' prefix and any non-alphabetic characters, keep any text following the first alphabetic character up to the '} suffix.
df["style"] = df["style"].str.extract(r"{'Format[^a-zA-Z]*([a-zA-Z].*?)'}")
styles = df["style"].value_counts(dropna=False)
# There's a few (small) categories that seem unlikely to be "digital music". Let's drop them.
drop_categories = ["Paperback", "Hardcover", "Kindle Edition", "USB Memory Stick", "Accessory", "Health and Beauty", "Calendar",
                   "Unknown Binding", "Spiral-bound", "Mass Market Paperback", "Kitchen", "Apparel", "Personal Computers",
                   "Office Product", "Grocery", "Unbound", "Audible Audiobook", "Perfect Paperback", "Misc. Supplies", "Home"]
df = df.loc[~df["style"].isin(drop_categories)]

# Don't need to check verified status unique values since we've already seen it's bool.
#endregion

#region DATA PROCESSING
# Convert review text to lower case.
df["reviewText"] = df["reviewText"].str.lower()
# Remove punctuation
df['reviewText'] = df['reviewText'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))


# Remove stop words.
def remove_stopwords(text):
    words = word_tokenize(text)
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)


stop_words = set(stopwords.words('english'))
df['reviewText'] = df['reviewText'].apply(remove_stopwords)

# Apply stemming to simplify text further.
stemmer = PorterStemmer()
df['stemmedText'] = df['reviewText'].apply(lambda x: stemmer.stem(x))
#endregion

# Pickle processed dataframe, so we don't have to rerun this script.
pickle_dir = 'processed data/processed_df.pickle'
export_path = os.path.join(current_dir, pickle_dir)
with open(export_path, 'wb') as f:
    pickle.dump(df, f)
