# Test script to play around with FiftyOne interface
# import matplotlib.pyplot as plt

import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.zoo as foz
import numpy as np
import pydicom
from PIL import Image

dataset = foz.load_zoo_dataset("mnist")
test_split = dataset.match_tags("test")  # only use the test split for now

brain_key = "mnist_test"

# image_files = test_split.values("filepath")
# embeddings = np.array([np.array(Image.open(f)).ravel() for f in image_files])
# results = compute_visualization(
#     test_split,
#     embeddings=embeddings,
#     num_dims=2,
#     method="tsne",
#     brain_key=brain_key,
#     verbose=True,
#     seed=51,
# )

results = dataset.load_brain_results(brain_key)  # load from pre-computed results

session = fo.launch_app(test_split)
session.wait()
