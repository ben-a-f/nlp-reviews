# This script performs some high-level exploratory data analysis (EDA) of our review dataset to inform our modelling choices.

import matplotlib
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle
import seaborn as sns

# Local issue with PyCharm and plotting. Comment out if not needed.
matplotlib.use("Qt5Agg")

#region DATA IMPORT
# Load pickled dataframe.
current_dir = os.path.dirname(os.path.abspath(__name__))
pickle_dir = 'processed data/processed_df.pickle'
export_path = os.path.join(current_dir, pickle_dir)
with open(export_path, 'rb') as f:
    df = pickle.load(f)
#endregion

#region EDA
# We'll create some plots for inspection.
# Score Distribution
sp = sns.displot(df, x="overall")
sp.set(xlabel="Review Score", ylabel="Count", title='Review Score Distribution')
# Low scores are much less common - we may need to make adjustments for unbalanced classes in our model (e.g. oversampling, or class weighting).

# Length Distribution
# Function to count words in a text
def count_words(text):
    words = text.split()
    return len(words)


df["reviewLength"] = df["stemmedText"].apply(count_words)
# Check for outliers at upper end.
df["reviewLength"].quantile([0.5, 0.99, 0.995, 1])
# There is a very small proportion of very long reviews.

# Review Length Plot
lp = sns.displot(df.loc[df["reviewLength"] <= 300], x="reviewLength", bins=20)
lp.set(xlabel="Review Length (Words)", ylabel="Count", title="Review Length Distribution")
# Most reviews are quite short.

# Length-Score Correlation
correlation = df["reviewLength"].corr(df["overall"])
# Negligible correlation; we don't need to worry too much about controlling for review length in the modelling.

# Verified vs Non Verified
vp = sns.FacetGrid(df, col="verified", height=5, aspect=1)
vp.map(sns.histplot, "overall", stat="percent", common_norm=False)
vp.set_titles("Verified = {col_name}")
vp.set_axis_labels("Score", "Proportion")
for ax in vp.axes.flat:
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha="center", va="bottom")
# Near-identical distributions, so we won't worry about treating non-verified reviews differently/excluding them.

#endregion
