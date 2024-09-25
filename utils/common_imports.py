"""Module to import common libraries and modules for the project."""

import json
import os
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import pylidc as pl
import seaborn as sns
from tqdm import tqdm

from project_config import config

# TODO move this to project_config.py
with open("pipeline_parameters.json", "r") as f:
    pipeline_config = json.load(f)


# type aliases for not breaking outdated pydicom numpy version:
np.int = int
np.float = float
