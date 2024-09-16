import os
from glob import glob

import numpy as np
import pydicom
import streamlit as st
from dotenv import load_dotenv

from project_config import config
from utils.utils import *
from visualisation_app.plotly_vis import plotly_fig2

load_dotenv(".env")
DATA_DIR = os.getenv("LIDC_IDRI_DIR")

st.set_page_config(page_title="3D Visualization", page_icon=":pill:", layout="wide")
st.title("3D Medical Imaging Visualisation App")
st.info(f"Found {len(config.patient_ids)} scans")
if selected_scan_id := st.selectbox("Select scan id to inspect", config.patient_ids):
    ct_slice_paths = get_ct_scan_slice_paths(selected_scan_id)
    st.info(f"Found {len(ct_slice_paths)} CT slices")
    dicom_files = [pydicom.dcmread(f) for f in sorted(ct_slice_paths)]
    ct_scan_data = np.stack([f.pixel_array for f in dicom_files], axis=0)
    st.write(ct_scan_data.shape)

    # st.image(normalize_image(images[slice_idx]), width=800)

    # st.plotly_chart(plotly_fig())

    st.plotly_chart(plotly_fig2(ct_scan_data), use_container_width=False)
