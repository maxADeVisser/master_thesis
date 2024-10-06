# Finding Hidden Stratifications in 3D medical data

## Setup

1. Install poetry (refer to [docs](https://python-poetry.org/docs/#installing-with-the-official-installer)):

   `curl -sSL https://install.python-poetry.org | python3 -`

2. Once installed, run `poetry install` which will install all dependencies specified in the `pyproject.toml` file

3. add a `.env` file in the root of the working directory and specify `LIDC_IDRI_DIR` as the directory path for the downloaded dataset

```mermaid
---
title: Nodule Dataframe Processing Flow
---

flowchart TD

0["`for each patient id`"]
A[fetch scan and accompanied annotations]
B[is there at least 1 annotation?]

0 --> A --> B
B --> |Yes|C
B -->|No|0

C[compute Consensus Centroid based on annotated segmentations]

subgraph compute_nodule_bbox
    direction LR
    D1[for each axis x,y,z in scan:]
    D2([extend the centroid with the half of the standardised img dim in both positive and negative direction])
    D3([does computed computed extension of centroid exceed the size of the original axis in the scan?])
    D4([Crop the nodule bbox so that it does not go over the edge of the scan])
    D5([continue])

    D1 --> D2
    D2 --> D3
    D3 -->|Yes|D4
    D4 --> D5
    D3 -->|No|D5
end

C --> compute_nodule_bbox

E[calculate malignancy scores and store other relevant info about the nodule]

compute_nodule_bbox --> E

F[consolidate retrieved/computed data attributes in a single dataframe entry]

E --> F

```
