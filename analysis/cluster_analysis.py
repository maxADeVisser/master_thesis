from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.io as pio
import seaborn as sns
import umap
from plotly.express import scatter_3d
from sklearn.manifold import TSNE

from project_config import SEED

# SCRIPT PARAMS ---------
fold = 3
experiment_id = "c30_3D_1711_1513"
hold_out = False
reduction_algo: Literal["umap", "tsne"] = "tsne"
# -----------------------

hold_indicator = "_holdout" if hold_out else ""
out = f"out/embeddings/{experiment_id}/fold{fold}"
model_embeddings = np.load(f"{out}/embeddings{hold_indicator}.npy")
labels = np.load(f"{out}/labels{hold_indicator}.npy")
print(model_embeddings.shape, labels.shape)

# Compute the 2D embeddings:
if reduction_algo == "tsne":
    tnse = TSNE(n_components=2, perplexity=30, random_state=SEED)
    dim_reduced_embeddings = tnse.fit_transform(model_embeddings)
elif reduction_algo == "umap":
    reducer = umap.UMAP(random_state=SEED)
    dim_reduced_embeddings = reducer.fit_transform(model_embeddings)

np.save(
    f"{out}/{reduction_algo}_embeddings{hold_indicator}.npy", dim_reduced_embeddings
)

# LOAD - these are 2D embeddings:
# dim_reduced_embeddings = np.load(
#     f"{out}/{reduction_algo}_embeddings{hold_indicator}.npy"
# )

tnse_df = pd.DataFrame.from_records(dim_reduced_embeddings)
tnse_df["label"] = labels

# plot the tnse embeddings
plt.figure(figsize=(10, 10))
sns.scatterplot(
    data=tnse_df,
    x=0,
    y=1,
    hue="label",
    palette="tab10",
    alpha=0.8,
    s=25,  # dot size
    # size="subtlety",
)
plt.legend(loc="upper right", title="True\nMalignancy\nScore")
plt.title(f"{reduction_algo} embeddings of Nodule ROIs")
plt.savefig(f"{out}/{reduction_algo}_embeddings_plot{hold_indicator}.png")
plt.axis("off")
plt.show()

# plot 3D
fig = scatter_3d(
    tnse_df,
    x=0,
    y=1,
    z=2,
    color="label",
    title="t-SNE embeddings",
    labels={"label": "Malignancy Score"},
    width=2000,
    height=1200,
)
# make the dot size smaller
fig.update_traces(marker=dict(size=4))
pio.write_html(
    fig,
    file=f"out/embeddings/{experiment_id}_fold{fold}/tnse_embeddings_plot{hold_indicator}.html",
)
fig.show()


# TODO
# evaluate clusters using silhouette score and other clustering metrics
# potentially use the elbow method to determine the optimal number of clusters
