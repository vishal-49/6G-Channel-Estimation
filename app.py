import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
import io
import os

from utils import load_model, predict


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="6G Channel Estimation",
    page_icon="📡",
    layout="wide"
)

# ==============================
# Sidebar
# ==============================

st.sidebar.title("📡 RDMSNet")

st.sidebar.markdown("---")

st.sidebar.subheader("Model Information")

st.sidebar.write("**Model:** RDMSNet")

st.sidebar.write("**Parameters:** 1,093,186")

st.sidebar.write("**Input Shape:** (2, 14, 612)")

st.sidebar.write("**Output Shape:** (2, 14, 612)")

import torch

if torch.cuda.is_available():
    device = "CUDA GPU"
elif torch.backends.mps.is_available():
    device = "Apple MPS"
else:
    device = "CPU"

st.sidebar.write(f"**Running On:** {device}")

st.sidebar.markdown("---")

st.sidebar.info(
    "Upload a 6G pilot signal (.npy). "
    "The trained RDMSNet model estimates the wireless channel."
)

st.title("📡 AI-Powered 6G Channel Estimation")

st.caption(
    "Residual Dense Multi-scale Squeeze-and-Excitation Network (RDMSNet)"
)

st.markdown("---")

st.write(
    """
Upload a **.npy** signal tensor of shape **(2, 14, 612)**.
The trained RDMSNet model will estimate the wireless channel.
"""
)

# --------------------------------------------------
# Load Model (Only Once)
# --------------------------------------------------

@st.cache_resource
def get_model():
    return load_model("model/Final_RDMSNet.pth")

model = get_model()

if st.button("🔄 Reset App"):
    st.rerun()

# --------------------------------------------------
# Upload
# --------------------------------------------------

# Upload File
# --------------------------------------------------
# Input Source Selection
# --------------------------------------------------

st.subheader("Input Source")

input_option = st.radio(
    "Choose Input Source",
    ["Use Built-in Sample", "Upload My Own File"],
    horizontal=True
)

signal = None

if input_option == "Use Built-in Sample":

    sample_choice = st.selectbox(
        "Select Sample",
        [
            "Sample 1",
            "Sample 2",
            "Sample 3"
        ]
    )

    sample_map = {
        "Sample 1": "samples/X_input_sample_0.npy",
        "Sample 2": "samples/X_input_sample_1.npy",
        "Sample 3": "samples/X_input_sample_2.npy"
    }

    sample_path = sample_map[sample_choice]

    signal = np.load(sample_path)

    st.success(f"Loaded {sample_choice}")

else:

    uploaded_file = st.file_uploader(
        "Upload Input Signal (.npy)",
        type=["npy"]
    )

    if uploaded_file is not None:

        signal = np.load(uploaded_file)

        st.success("Input file loaded successfully.")

if signal is not None:

    try:

        st.success("Input file loaded successfully.")

        st.write("### Input Information")

        st.write(f"Shape : {signal.shape}")
        st.write(f"Datatype : {signal.dtype}")

        st.subheader("Input Signal Statistics")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Minimum", f"{signal.min():.4f}")
            st.metric("Maximum", f"{signal.max():.4f}")

        with col2:
            st.metric("Mean", f"{signal.mean():.4f}")
            st.metric("Std", f"{signal.std():.4f}")

        if signal.shape != (2, 14, 612):

            st.error(
                "Invalid input shape.\nExpected (2,14,612)"
            )

        else:

            if st.button("Estimate Channel"):

                with st.spinner("Running RDMSNet..."):

                    start = time.time()

                    prediction = predict(model, signal)

                    end = time.time()

                inference_time = end - start

                st.success("✅ Channel Estimation Completed!")

                st.info(f"⏱ Inference Time: {inference_time:.4f} seconds")

                st.write("### Estimated Channel")

                st.write(f"Output Shape : {prediction.shape}")

                st.subheader("Estimated Channel Statistics")

                col1, col2 = st.columns(2)

                with col1:

                    st.metric("Minimum", f"{prediction.min():.4f}")

                    st.metric("Maximum", f"{prediction.max():.4f}")

                with col2:

                    st.metric("Mean", f"{prediction.mean():.4f}")

                    st.metric("Std", f"{prediction.std():.4f}")

                st.subheader("Input Signal Visualization")

                input_real = signal[0]
                input_imag = signal[1]

                col1, col2 = st.columns(2)

                with col1:

                    fig1, ax1 = plt.subplots(figsize=(8,4))

                    im = ax1.imshow(
                        input_real,
                        aspect="auto",
                        origin="lower"
                    )

                    ax1.set_title("Input - Real Part")

                    ax1.set_xlabel("Subcarriers")

                    ax1.set_ylabel("OFDM Symbols")

                    plt.colorbar(im)

                    st.pyplot(fig1)

                with col2:

                    fig2, ax2 = plt.subplots(figsize=(8,4))

                    im = ax2.imshow(
                        input_imag,
                        aspect="auto",
                        origin="lower"
                    )

                    ax2.set_title("Input - Imaginary Part")

                    ax2.set_xlabel("Subcarriers")

                    ax2.set_ylabel("OFDM Symbols")

                    plt.colorbar(im)

                    st.pyplot(fig2)

                # -----------------------------
                # Heatmaps
                # -----------------------------

                st.subheader("Estimated Channel Visualization")

                real = prediction[0]
                imag = prediction[1]

                col1, col2 = st.columns(2)

                with col1:

                    fig1, ax1 = plt.subplots(figsize=(8,4))

                    im = ax1.imshow(
                        real,
                        aspect="auto",
                        origin="lower"
                    )

                    ax1.set_title("Estimated - Real Part")
                    ax1.set_xlabel("Subcarriers")
                    ax1.set_ylabel("OFDM Symbols")

                    plt.colorbar(im)

                    st.pyplot(fig1)

                with col2:

                    fig2, ax2 = plt.subplots(figsize=(8,4))

                    im = ax2.imshow(
                        imag,
                        aspect="auto",
                        origin="lower"
                    )

                    ax2.set_title("Estimated - Imaginary Part")
                    ax2.set_xlabel("Subcarriers")
                    ax2.set_ylabel("OFDM Symbols")

                    plt.colorbar(im)

                    st.pyplot(fig2)

                # -----------------------------
                # Download
                # -----------------------------

                buffer = io.BytesIO()               

                np.save(buffer, prediction)

                buffer.seek(0)

                st.download_button(
                    label="⬇ Download Estimated Channel (.npy)",
                    data=buffer,
                    file_name="Estimated_Channel.npy",
                    mime="application/octet-stream"
                )

    except Exception as e:

        st.error(str(e))


st.markdown("---")

with st.expander("ℹ️ About RDMSNet"):

    st.markdown("""
### RDMSNet Architecture

RDMSNet (Residual Dense Multi-scale Squeeze-and-Excitation Network)
is a deep learning model developed for 6G wireless channel estimation.

### Model Details

- **Model:** RDMSNet
- **Framework:** PyTorch
- **Total Parameters:** 1,093,186
- **Input Shape:** (2, 14, 612)
- **Output Shape:** (2, 14, 612)

### Dataset Information

- Samples: 10,000
- Carrier Frequency: 7 GHz
- OFDM Symbols: 14
- Subcarriers: 612

### Channel Profiles

- TDL-A
- TDL-B
- TDL-C
- TDL-D
- TDL-E

### Supported SNR

-10 dB to +30 dB
""")

st.markdown("---")

st.caption(
    "Developed for 6G Wireless Channel Estimation using RDMSNet | "
    "Deep Learning • PyTorch • Streamlit"
)