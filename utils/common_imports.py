"""Module to import common libraries and modules for the project."""

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

# type aliases for not breaking outdated pydicom numpy version:
np.int = int
np.float = float