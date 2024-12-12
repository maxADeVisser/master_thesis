
import pandas as pd

from google.colab import files
uploaded = files.upload()

# Load the file from the uploaded dictionary
import pandas as pd
data = pd.read_csv(list(uploaded.keys())[0])

# Check the data
print(data.head())

# We need to transpose the data to have raters as rows and items (nodules) as columns
data_t = data.T

# Calculate Kendall's W - Items and raters
n_items = data.shape[0]  # number of items (nodules)
n_raters = data.shape[1]  # number of raters (annotations)

# Rank the ratings for each subject
ranked_data = data_t.rank(axis=1)

# Compute the sum of ranks for each item
sum_of_ranks = ranked_data.sum(axis=0)

# Compute the mean of the sum of ranks
mean_of_ranks = sum_of_ranks.mean()

# Compute the sum of squared deviations from the mean rank sum
S = sum((sum_of_ranks - mean_of_ranks)**2)

#  Calculate Kendall's W
W = (12 * S) / (n_raters**2 * (n_items**3 - n_items))

# Output the result
print(f"Kendall's W (Coefficient of Concordance): {W}")