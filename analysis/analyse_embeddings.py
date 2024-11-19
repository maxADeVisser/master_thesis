import matplotlib.pyplot as plt
import pandas as pd
import plotly.io as pio
import seaborn as sns
from plotly.express import scatter_3d

from project_config import env_config

# SCRIPT PARAMS ---------
# TODO remove TNSE stuff from here. this should be in the get_embeddings.py script
fold = 3
experiment_id = "c30_3D_1711_1513"
hold_out = False
# -----------------------

hold_indicator = "_holdout" if hold_out else ""
out = f"{env_config.OUT_DIR}/embeddings/{experiment_id}/fold{fold}"

# LOAD - these are 2D embeddings:
dim_reduced_embeddings = pd.read_csv(
    "out/embeddings/c30_3D_1711_1513/fold3/embeddings_df.csv"
).query("dataset == 'train'")

# plot the tnse embeddings
plt.figure(figsize=(10, 10))
sns.scatterplot(
    data=dim_reduced_embeddings,
    x="x_embed",
    y="y_embed",
    hue="label",
    palette="tab10",
    alpha=0.8,
    s=25,  # dot size
    # size="subtlety",
)
plt.legend(loc="upper right", title="True\nMalignancy\nScore")
plt.title(f"tnse embeddings of Nodule ROIs")
plt.savefig(f"{out}/tnse_embeddings_plot{hold_indicator}.png")
plt.axis("off")
plt.show()

# plot 3D
fig = scatter_3d(
    dim_reduced_embeddings,
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


# TODO ???
# evaluate clusters using silhouette score and other clustering metrics
# potentially use the elbow method to determine the optimal number of clusters
