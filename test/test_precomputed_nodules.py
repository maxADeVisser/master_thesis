import os
import re
from glob import glob

import torch
from dotenv import load_dotenv

load_dotenv()

PROJECT_DIR = os.getenv("PROJECT_DIR")
DATA_DIR = f"{PROJECT_DIR}/data"

all_dirs = glob(f"{DATA_DIR}/precomputed*")

context_regex = r"(\d+)(?=C)"
dimension_regex = r"(\d+(\.\d+)?)(?=D)"
for d in all_dirs:
    context = re.search(context_regex, d).group()
    dimension = re.search(dimension_regex, d).group()
    if context and dimension:
        # load the a scan in the directory
        scan = torch.load(glob(f"{d}/*.pt")[0], weights_only=True)
        if dimension == 2.5:
            assert scan[0].shape == (
                3,
                context,
                context,
                context,
            ), "Incorrect shape for 2.5D ROI"
        elif dimension == 3:
            assert scan[0].shape == (
                1,
                context,
                context,
                context,
            ), "Incorrect shape for 3D ROI"
print("All precomputed ROIs have the correct shape.")
