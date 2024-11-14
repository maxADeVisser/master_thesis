import numpy as np
from sklearn.manifold import TSNE

from model.MEDMnist.ResNet import load_resnet_model
from project_config import SEED

fold = 0
embeddings_path = f"out/embeddings/fold{fold}/embeddings.npy"
embeddings = np.load(embeddings_path)
tnse_embeddings_out = f"out/embeddings/fold{fold}/tnse_embeddings.npy"

tnse = TSNE(n_components=2, random_state=SEED)
tnse_embeddings = tnse.fit_transform(embeddings)
np.save(tnse_embeddings_out, tnse_embeddings)

# plot the tnse embeddings
