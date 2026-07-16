import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import io
import torch

from utils import load_model, predict

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="6G Channel Estimation | Research Demonstration Platform",
    page_icon="📡",
    layout="wide"
)

# --------------------------------------------------
# WCAG-Compliant Light Theme Academic CSS (Minified)
# --------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        html, body, [data-testid="stAppViewContainer"], .main { font-family: 'Inter', sans-serif !important; background-color: #F8FAFC !important; color: #111827 !important; }
        .platform-title { font-size: 2.2rem; font-weight: 700; color: #111827 !important; margin-bottom: 0.1rem; text-align: left; }
        .platform-subtitle { font-size: 1rem; color: #4B5563 !important; margin-top: 0; margin-bottom: 1.5rem; text-transform: uppercase; letter-spacing: 1.2px; font-weight: 600; text-align: left; }
        h1, h2, h3, h4, h5, h6 { color: #111827 !important; font-family: 'Inter', sans-serif !important; font-weight: 600 !important; margin-top: 1.8rem !important; margin-bottom: 0.8rem !important; text-align: left !important; }
        [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li, [data-testid="stMarkdownContainer"] span, [data-testid="stMarkdownContainer"] div { color: #111827 !important; }
        .section-divider { border-top: 1px solid #E5E7EB; margin: 2rem 0; }
        [data-testid="stSidebar"] { background-color: #FFFFFF !important; border-right: 1px solid #E5E7EB !important; }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p, [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span, [data-testid="stSidebar"] label { color: #111827 !important; }
        div[data-testid="stMetric"] { background-color: #FFFFFF !important; border: 1px solid #E5E7EB !important; border-radius: 4px; padding: 14px 18px; box-shadow: none !important; }
        div[data-testid="stMetric"] label { color: #4B5563 !important; font-weight: 600 !important; }
        div[data-testid="stMetricValue"] > div { color: #2563EB !important; font-weight: 700 !important; }
        div[data-testid="stTable"] table { background-color: #FFFFFF !important; border: 1px solid #E5E7EB !important; }
        div[data-testid="stTable"] th, div[data-testid="stTable"] thead tr th { background-color: #F8FAFC !important; color: #111827 !important; font-weight: 600 !important; border-bottom: 1px solid #E5E7EB !important; }
        div[data-testid="stTable"] td, div[data-testid="stTable"] tbody tr td { background-color: #FFFFFF !important; color: #4B5563 !important; border-bottom: 1px solid #E5E7EB !important; }
        div.stButton > button { background-color: #FFFFFF !important; color: #111827 !important; border: 1px solid #E5E7EB !important; border-radius: 4px !important; padding: 8px 24px !important; font-weight: 500 !important; transition: all 0.15s ease !important; }
        div.stButton > button:hover { background-color: #F8FAFC !important; border-color: #cbd5e1 !important; color: #2563EB !important; }
        div.stButton > button[data-testid="stBaseButton-primary"] { background-color: #2563EB !important; border: 1px solid #2563EB !important; }
        div.stButton > button[data-testid="stBaseButton-primary"] * { color: #FFFFFF !important; }
        div.stButton > button[data-testid="stBaseButton-primary"]:hover { background-color: #1d4ed8 !important; border-color: #1d4ed8 !important; }
        div.stDownloadButton > button { background-color: #2563EB !important; border: 1px solid #2563EB !important; border-radius: 4px !important; padding: 8px 24px !important; font-weight: 500 !important; }
        div.stDownloadButton > button * { color: #FFFFFF !important; }
        div.stDownloadButton > button:hover { background-color: #1d4ed8 !important; border-color: #1d4ed8 !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# Device Detection & State Management
# --------------------------------------------------
device = "CUDA GPU" if torch.cuda.is_available() else ("Apple MPS" if torch.backends.mps.is_available() else "CPU")

for key in ["signal", "prediction", "inference_time", "last_sample_choice", "last_uploaded_file", "last_input_option"]:
    if key not in st.session_state:
        st.session_state[key] = None

# --------------------------------------------------
# Minimal Sidebar
# --------------------------------------------------
st.sidebar.markdown('### System Panel')
st.sidebar.markdown(f"**Execution Device:** {device}")

if st.session_state.signal is None:
    st.sidebar.warning("Waiting for Input Data")
elif st.session_state.prediction is None:
    st.sidebar.info("Ready for Inference")
else:
    st.sidebar.success("Inference Completed")

st.sidebar.markdown("---")

if st.sidebar.button("Reset Application", use_container_width=True):
    for key in st.session_state.keys():
        st.session_state[key] = None
    st.rerun()

# --------------------------------------------------
# Header & About the Framework
# --------------------------------------------------
st.markdown('<div class="platform-title">6G Channel Estimation</div>', unsafe_allow_html=True)
st.markdown('<div class="platform-subtitle">Research Demonstration Platform</div><div class="section-divider"></div>', unsafe_allow_html=True)

st.header("About the Framework")
st.markdown(
    r"""
The **Residual Dense Multi-scale Squeeze-and-Excitation Network (RDMSNet)** is a deep learning architecture 
specifically developed for **6G wireless channel estimation**. Fast-varying channels in high-frequency bands 
require high-resolution Channel State Information (CSI) interpolation. RDMSNet addresses this by framing pilot 
signal grids as complex 2D spatial tensors and recovering complete channel responses.

The physical channel observation model is formulated as:
$$y = H x + n$$
where $y$ is the received pilot signal, $H$ is the high-fidelity complex channel response, $x$ is the known pilot grid configuration, and $n$ represents additive white Gaussian noise (AWGN) and inter-carrier interference.
"""
)
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# --------------------------------------------------
# Load Model & Input Selection
# --------------------------------------------------
@st.cache_resource
def get_model():
    return load_model("model/Final_RDMSNet.pth")

model = get_model()

metadata = pd.read_csv("samples/sample_metadata.csv")

st.header("Upload Channel Data")
st.write(
    "Provide a pilot signal tensor file in the standard NumPy binary (.npy) format. "
    "Expected shape is exactly (2, 14, 612), mapping to real/imaginary parts, 14 OFDM symbols, and 612 subcarriers."
)

input_option = st.radio("Select Data Source", ["Use Built-in Sample", "Upload My Own File"], horizontal=True, label_visibility="collapsed")
signal = None

if st.session_state.last_input_option != input_option:
    st.session_state.signal = st.session_state.prediction = st.session_state.inference_time = st.session_state.last_sample_choice = st.session_state.last_uploaded_file = None
    st.session_state.last_input_option = input_option
    st.rerun()

if input_option == "Use Built-in Sample":
    import os

    sample_files = sorted(
        [
            f
            for f in os.listdir("samples")
            if f.startswith("X_input_sample_")
            and f.endswith(".npy")
        ],
        key=lambda x: int(x.split("_")[-1].replace(".npy", ""))
    )

    sample_names = [
        f"Sample {i+1}"
        for i in range(len(sample_files))
    ]

    sample_choice = st.selectbox(
        "Select Built-in Sample",
        [f"Sample {i}" for i in range(1, 101)]
    )

    sample_idx = sample_names.index(sample_choice)

    row = metadata.iloc[sample_idx]

    st.markdown("#### Sample Information")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Delay Profile", row["delay_profile"])
    c2.metric("SNR (dB)", f"{row['snr']:.1f}")
    c3.metric("Speed (m/s)", f"{row['speed']:.1f}")
    c4.metric("Doppler (Hz)", f"{row['doppler']:.0f}")  

    sample_path = os.path.join(
        "samples",
        sample_files[sample_idx]
    )
    
    try:
        signal = np.load(sample_path)
        st.session_state.signal = signal
        if st.session_state.last_sample_choice != sample_choice:
            st.session_state.prediction = st.session_state.inference_time = None
            st.session_state.last_sample_choice = sample_choice
            st.rerun()
    except Exception as e:
        st.error(f"Error loading built-in sample: {e}")
else:
    uploaded_file = st.file_uploader("Upload NumPy File (.npy)", type=["npy"], label_visibility="collapsed")
    if uploaded_file is not None:
        try:
            signal = np.load(uploaded_file)
            st.session_state.signal = signal
            if st.session_state.last_uploaded_file != uploaded_file.name:
                st.session_state.prediction = st.session_state.inference_time = None
                st.session_state.last_uploaded_file = uploaded_file.name
                st.rerun()
        except Exception as e:
            st.error(f"Error reading file: {e}")

# Display Statistics (Using Native Streamlit Table)
if st.session_state.signal is not None:
    sig = st.session_state.signal
    st.write("**Loaded Signal Characteristics & Metrics:**")
    st.table({
        "Parameter": ["Tensor Dimension", "Data Type", "Minimum Value", "Maximum Value", "Mean", "Standard Deviation"],
        "Value": [str(sig.shape), str(sig.dtype), f"{sig.min():.5f}", f"{sig.max():.5f}", f"{sig.mean():.5f}", f"{sig.std():.5f}"]
    })
    if sig.shape != (2, 14, 612):
        st.error(f"Error: Invalid signal shape {sig.shape}. Model expects shape (2, 14, 612).")
else:
    st.info("No signal loaded. Select a built-in sample or upload a custom file.")
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# --------------------------------------------------
# Model Information & Inference Execution
# --------------------------------------------------
st.header("Model Information")
st.table({
    "System Attribute": ["Model Architecture", "Framework & Weights", "Model Parameters", "Input / Output Shapes", "Active Execution Engine"],
    "Specifications / Details": ["RDMSNet (Residual Dense Multi-scale Squeeze-and-Excitation Network)", "PyTorch (model weights loaded from model/Final_RDMSNet.pth)", "1,093,186 trainable parameters", "(2, 14, 612) (2 Channels: Real/Imag, 14 OFDM Symbols, 612 Subcarriers)", device]
})
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

st.header("Run Inference")
if st.session_state.signal is None:
    st.warning("Please upload or select an input signal under 'Upload Channel Data' to enable inference.")
elif st.session_state.signal.shape != (2, 14, 612):
    st.error("Inference disabled due to invalid tensor shape.")
else:
    st.write("Click below to process the pilot signal through RDMSNet.")
    if st.button("Run Channel Estimation", type="primary"):
        with st.spinner("Processing deep learning channel estimation..."):
            start = time.time()
            st.session_state.prediction = predict(model, st.session_state.signal)
            st.session_state.inference_time = time.time() - start
            st.rerun()
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# --------------------------------------------------
# Prediction Summary
# --------------------------------------------------
st.header("Prediction Summary")
if st.session_state.prediction is not None:
    pred = st.session_state.prediction
    st.table({
        "Metric Parameter": ["Execution Status", "Model Inference Speed", "Output Tensor Dimension", "Minimum Value", "Maximum Value", "Mean", "Standard Deviation"],
        "Value / Measurement": ["Success (Inference Completed)", f"{st.session_state.inference_time:.5f} seconds", str(pred.shape), f"{pred.min():.5f}", f"{pred.max():.5f}", f"{pred.mean():.5f}", f"{pred.std():.5f}"]
    })
else:
    st.info("Inference output is empty. Run inference to generate telemetry.")
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# --------------------------------------------------
# Visualization
# --------------------------------------------------
st.header("Visualization")

def plot_channel_grid(real_data, imag_data, title_suffix):
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, data, comp, cmap in zip(axs, [real_data, imag_data], ["Real", "Imaginary"], ["viridis", "plasma"]):
        im = ax.imshow(data, aspect="auto", origin="lower", cmap=cmap)
        ax.set_title(f"{title_suffix} - {comp} Component", fontsize=9, fontweight="bold", pad=6)
        ax.set_xlabel("Subcarrier Index", fontsize=8)
        ax.set_ylabel("OFDM Symbol Index", fontsize=8)
        ax.tick_params(labelsize=8)
        fig.colorbar(im, ax=ax)
    plt.tight_layout()
    st.pyplot(fig)

if st.session_state.signal is None:
    st.info("No signal loaded. Visualizations are unavailable.")
else:
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'axes.edgecolor': '#E5E7EB', 'axes.linewidth': 0.8,
        'xtick.color': '#4B5563', 'ytick.color': '#4B5563', 'text.color': '#111827'
    })
    
    if st.session_state.prediction is None:
        st.write("Showing 2D channel profiles for Input Signal:")
        plot_channel_grid(st.session_state.signal[0], st.session_state.signal[1], "Input Signal")
    else:
        st.write("Showing 2D comparison between Input Pilot Grid and Estimated Channel Response:")
        plot_channel_grid(st.session_state.signal[0], st.session_state.signal[1], "Input Pilot Grid")
        plot_channel_grid(st.session_state.prediction[0], st.session_state.prediction[1], "RDMSNet Estimated Response")

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# --------------------------------------------------
# Download Results & Footer
# --------------------------------------------------
st.header("Download Results")
if st.session_state.prediction is not None:
    st.write("Export and download the reconstructed 6G channel response as a NumPy array file.")
    buffer = io.BytesIO()
    np.save(buffer, st.session_state.prediction)
    buffer.seek(0)
    st.download_button(label="Download Estimated Channel (.npy)", data=buffer, file_name="Estimated_Channel.npy", mime="application/octet-stream")
else:
    st.info("Estimated tensor data is currently unavailable.")

st.markdown(
    """
    <hr style="margin-top: 50px; margin-bottom: 15px; border: 0; border-top: 1px solid #E5E7EB;">
    <div style="text-align: center; font-size: 0.8rem; color: #6B7280; padding-bottom: 20px;">
        <p style="margin: 4px 0;">RDMSNet Platform for 6G Wireless Channel Estimation | PyTorch Neural Network Implementation</p>
        <p style="margin: 4px 0;">Version 5.0 • Academic Research Demo Tool</p>
    </div>
    """,
    unsafe_allow_html=True
)
