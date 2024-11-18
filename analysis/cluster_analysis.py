import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.io as pio
import seaborn as sns
from plotly.express import scatter_3d
from sklearn.manifold import TSNE

from project_config import SEED

fold = 3
experiment_id = "c30_3D_1711_1513"
out = f"out/embeddings/{experiment_id}/fold{fold}"
embeddings = np.load(f"{out}/embeddings.npy")
labels = np.load(f"{out}/labels.npy")
print(embeddings.shape, labels.shape)

tnse = TSNE(n_components=2, perplexity=30, random_state=SEED)
tnse_embeddings = tnse.fit_transform(embeddings)
np.save(f"{out}/tnse_embeddings.npy", tnse_embeddings)

# tnse_embeddings = np.load(tnse_embeddings_out)  # these are 2D embeddings
tnse_df = pd.DataFrame.from_records(tnse_embeddings)
tnse_df["label"] = labels
tnse_df

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
plt.title("t-SNE embeddings of Nodule ROIs")
plt.savefig(f"{out}/tnse_embeddings_plot.png")
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
    fig, file=f"out/embeddings/{experiment_id}_fold{fold}/tnse_embeddings_plot.html"
)
fig.show()


# TODO
# evaluate clusters using silhouette score and other clustering metrics
# potentially use the elbow method to determine the optimal number of clusters
