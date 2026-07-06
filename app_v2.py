import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
import io
import os
import torch

from utils import load_model, predict

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="6G Channel Estimation | RDMSNet Dashboard",
    page_icon="📡",
    layout="wide"
)

# --------------------------------------------------
# Custom CSS for Modern AI Dashboard Styling
# --------------------------------------------------
st.markdown(
    """
    <style>
        /* Custom Font and spacing overrides */
        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 2rem !important;
        }
        
        /* Metric Card Styling */
        div[data-testid="stMetric"] {
            background-color: var(--secondary-background-color);
            border: 1px solid rgba(128, 128, 128, 0.15);
            border-radius: 12px;
            padding: 16px 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.03);
            transition: all 0.3s ease;
        }
        div[data-testid="stMetric"]:hover {
            border-color: #6366f1;
            transform: translateY(-2px);
            box-shadow: 0 8px 16px rgba(99, 102, 241, 0.1);
        }
        
        /* Tab Button Styling */
        button[data-baseweb="tab"] {
            font-size: 1.05rem !important;
            font-weight: 600 !important;
            padding: 10px 20px !important;
            border-radius: 6px 6px 0 0 !important;
            transition: all 0.2s ease !important;
        }
        button[data-baseweb="tab"]:hover {
            color: #6366f1 !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            background-color: rgba(99, 102, 241, 0.08) !important;
            color: #6366f1 !important;
            border-bottom: 3px solid #6366f1 !important;
        }
        
        /* Primary/Inference buttons styling */
        div.stButton > button {
            background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 10px 24px !important;
            font-weight: 600 !important;
            box-shadow: 0 4px 6px rgba(99, 102, 241, 0.2) !important;
            transition: all 0.2s ease !important;
            width: 100% !important;
        }
        div.stButton > button:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 6px 12px rgba(99, 102, 241, 0.3) !important;
            background: linear-gradient(135deg, #4338ca 0%, #4f46e5 100%) !important;
        }
        div.stButton > button:active {
            transform: translateY(1px) !important;
        }
        
        /* Sidebar Reset button override */
        section[data-testid="stSidebar"] div.stButton > button {
            background: transparent !important;
            color: var(--text-color) !important;
            border: 1px solid rgba(128, 128, 128, 0.4) !important;
            box-shadow: none !important;
            font-weight: 500 !important;
            padding: 8px 16px !important;
        }
        section[data-testid="stSidebar"] div.stButton > button:hover {
            background: rgba(128, 128, 128, 0.1) !important;
            color: #6366f1 !important;
            border-color: #6366f1 !important;
            transform: none !important;
            box-shadow: none !important;
        }
        
        /* Download button styling */
        div.stDownloadButton > button {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 10px 24px !important;
            font-weight: 600 !important;
            box-shadow: 0 4px 6px rgba(16, 185, 129, 0.2) !important;
            transition: all 0.2s ease !important;
            width: 100% !important;
        }
        div.stDownloadButton > button:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 6px 12px rgba(16, 185, 129, 0.3) !important;
            background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
        }
        div.stDownloadButton > button:active {
            transform: translateY(1px) !important;
        }
        
        /* Dashboard Card Container */
        .dashboard-card {
            background-color: var(--secondary-background-color);
            border: 1px solid rgba(128, 128, 128, 0.15);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.01);
        }
        
        /* Sidebar Card Container */
        .sidebar-card {
            background-color: var(--secondary-background-color);
            border: 1px solid rgba(128, 128, 128, 0.15);
            border-radius: 10px;
            padding: 14px;
            margin-bottom: 14px;
            font-size: 0.9rem;
            line-height: 1.5;
        }
        .sidebar-card h4 {
            margin-top: 0;
            margin-bottom: 8px;
            font-size: 0.95rem;
            color: #6366f1;
            font-weight: 600;
        }
        .sidebar-card p {
            margin: 4px 0;
        }
        
        /* Expander Styling */
        .streamlit-expanderHeader {
            background-color: var(--secondary-background-color) !important;
            border: 1px solid rgba(128, 128, 128, 0.15) !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
        }
        
        /* Radio Layout Styling */
        div[data-testid="stRadio"] > label {
            font-weight: 600 !important;
            margin-bottom: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# Device Detection
# --------------------------------------------------
if torch.cuda.is_available():
    device = "CUDA GPU"
    device_icon = "🟢"
elif torch.backends.mps.is_available():
    device = "Apple MPS"
    device_icon = "🟢"
else:
    device = "CPU"
    device_icon = "⚪"

# --------------------------------------------------
# Session State Initialization
# --------------------------------------------------
if "signal" not in st.session_state:
    st.session_state.signal = None
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "inference_time" not in st.session_state:
    st.session_state.inference_time = None
if "last_sample_choice" not in st.session_state:
    st.session_state.last_sample_choice = None
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None
if "last_input_option" not in st.session_state:
    st.session_state.last_input_option = None

# --------------------------------------------------
# Sidebar Redesign
# --------------------------------------------------
st.sidebar.markdown(
    """
    <div style="display: flex; align-items: center; gap: 10px; padding: 10px 0;">
        <span style="font-size: 1.8rem;">📡</span>
        <h2 style="margin: 0; font-size: 1.5rem; font-weight: 700;">RDMSNet</h2>
    </div>
    <hr style="margin: 8px 0 20px 0; border: 0; border-top: 1px solid rgba(128, 128, 128, 0.2);">
    """,
    unsafe_allow_html=True
)

# Model Information Container
st.sidebar.markdown(
    """
    <div class="sidebar-card">
        <h4>🤖 Model Details</h4>
        <p><b>Model:</b> RDMSNet</p>
        <p><b>Framework:</b> PyTorch</p>
        <p><b>Parameters:</b> 1,093,186</p>
        <p><b>Input Shape:</b> (2, 14, 612)</p>
        <p><b>Output Shape:</b> (2, 14, 612)</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Running Device Container
st.sidebar.markdown(
    f"""
    <div class="sidebar-card">
        <h4>⚙️ Environment</h4>
        <p><b>Device:</b> {device_icon} {device}</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Status Indicator
if st.session_state.signal is None:
    status_text = "⏳ Waiting for Input"
    status_style = "color: #ef4444;"
elif st.session_state.prediction is None:
    status_text = "⚡ Ready to Estimate"
    status_style = "color: #f59e0b;"
else:
    status_text = "✅ Estimation Complete"
    status_style = "color: #10b981;"

st.sidebar.markdown(
    f"""
    <div class="sidebar-card">
        <h4>📊 System Status</h4>
        <p><b>Status:</b> <span style="{status_style} font-weight: 600;">{status_text}</span></p>
    </div>
    """,
    unsafe_allow_html=True
)

# Reset Button in Sidebar
st.sidebar.markdown("<br>", unsafe_allow_html=True)
if st.sidebar.button("🔄 Reset Application"):
    st.session_state.signal = None
    st.session_state.prediction = None
    st.session_state.inference_time = None
    st.session_state.last_sample_choice = None
    st.session_state.last_uploaded_file = None
    st.session_state.last_input_option = None
    st.rerun()

# --------------------------------------------------
# Main Hero Header
# --------------------------------------------------
st.markdown(
    """
    <div style="text-align: center; padding: 15px 0 5px 0;">
        <h1 style="font-size: 2.3rem; font-weight: 800; margin-bottom: 6px;">📡 AI-Powered 6G Wireless Channel Estimation</h1>
        <h3 style="font-size: 1.2rem; font-weight: 500; color: #6366f1; margin-top: 0; margin-bottom: 12px;">
            Residual Dense Multi-scale Squeeze-and-Excitation Network (RDMSNet)
        </h3>
        <p style="font-size: 0.95rem; color: #888; max-width: 820px; margin: 0 auto; line-height: 1.6;">
            A deep learning-based research tool for reconstructing complete wireless channel responses from pilot signals. 
            Select or upload a pilot tensor of shape (2, 14, 612) below to begin.
        </p>
    </div>
    <hr style="margin-top: 15px; margin-bottom: 25px; border: 0; border-top: 1px solid rgba(128,128,128,0.2);">
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# Load Model (Once)
# --------------------------------------------------
@st.cache_resource
def get_model():
    return load_model("model/Final_RDMSNet.pth")

model = get_model()

# --------------------------------------------------
# Dashboard Tabs
# --------------------------------------------------
tab_input, tab_est, tab_viz, tab_about = st.tabs([
    "📡 Input Signal Setup",
    "🔮 Channel Estimation",
    "📊 Profile Visualizations",
    "ℹ️ About RDMSNet"
])

# --------------------------------------------------
# TAB 1: Input Signal Selection & Statistics
# --------------------------------------------------
with tab_input:
    st.markdown("### 🛠️ Input Source Settings")
    
    input_option = st.radio(
        "Choose Input Source",
        ["Use Built-in Sample", "Upload My Own File"],
        horizontal=True
    )
    
    signal = None
    
    if input_option == "Use Built-in Sample":
        sample_choice = st.selectbox(
            "Select Built-in Sample",
            ["Sample 1", "Sample 2", "Sample 3"]
        )
        sample_map = {
            "Sample 1": "samples/X_input_sample_0.npy",
            "Sample 2": "samples/X_input_sample_1.npy",
            "Sample 3": "samples/X_input_sample_2.npy"
        }
        sample_path = sample_map[sample_choice]
        
        try:
            signal = np.load(sample_path)
            st.session_state.signal = signal
            
            # Detect changes to clear prediction
            if (st.session_state.last_sample_choice != sample_choice or 
                st.session_state.last_input_option != input_option):
                st.session_state.prediction = None
                st.session_state.inference_time = None
                st.session_state.last_sample_choice = sample_choice
                st.session_state.last_input_option = input_option
                st.rerun()
                
            st.success(f"✅ Loaded {sample_choice}")
        except Exception as e:
            st.error(f"Error loading built-in sample: {str(e)}")
            
    else:
        uploaded_file = st.file_uploader(
            "Upload Input Signal (.npy)",
            type=["npy"]
        )
        
        if uploaded_file is not None:
            try:
                signal = np.load(uploaded_file)
                st.session_state.signal = signal
                
                # Detect changes to clear prediction
                if (st.session_state.last_uploaded_file != uploaded_file.name or 
                    st.session_state.last_input_option != input_option):
                    st.session_state.prediction = None
                    st.session_state.inference_time = None
                    st.session_state.last_uploaded_file = uploaded_file.name
                    st.session_state.last_input_option = input_option
                    st.rerun()
                    
                st.success("✅ Input file loaded successfully.")
            except Exception as e:
                st.error(f"Error reading uploaded file: {str(e)}")
        else:
            if st.session_state.last_input_option != input_option:
                st.session_state.signal = None
                st.session_state.prediction = None
                st.session_state.inference_time = None
                st.session_state.last_input_option = input_option
                st.rerun()

    # Display Input Info & Statistics
    if st.session_state.signal is not None:
        signal = st.session_state.signal
        
        st.markdown("---")
        st.markdown("### 📋 Input Tensor Information")
        
        col_meta1, col_meta2 = st.columns(2)
        with col_meta1:
            st.markdown(
                f"""
                <div class="dashboard-card">
                    <p style="margin: 0; font-size: 0.9rem; color: #888; font-weight: 500;">Tensor Dimension Shape</p>
                    <h3 style="margin: 4px 0 0 0; font-weight: 700; color: #6366f1;">{signal.shape}</h3>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col_meta2:
            st.markdown(
                f"""
                <div class="dashboard-card">
                    <p style="margin: 0; font-size: 0.9rem; color: #888; font-weight: 500;">Data Type Precision</p>
                    <h3 style="margin: 4px 0 0 0; font-weight: 700; color: #6366f1;">{signal.dtype}</h3>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        st.markdown("### 📊 Input Signal Statistics")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Minimum", f"{signal.min():.4f}")
        with c2:
            st.metric("Maximum", f"{signal.max():.4f}")
        with c3:
            st.metric("Mean", f"{signal.mean():.4f}")
        with c4:
            st.metric("Std", f"{signal.std():.4f}")
            
        if signal.shape != (2, 14, 612):
            st.error("❌ Invalid input shape. Expected dimensions are exactly (2, 14, 612).")
        else:
            st.markdown("<br>", unsafe_allow_html=True)
            st.info("💡 Signal is correctly formatted. Switch to the **Channel Estimation** tab to run inference.")

# --------------------------------------------------
# TAB 2: Model Inference & Performance
# --------------------------------------------------
with tab_est:
    st.markdown("### 🔮 RDMSNet Estimation Engine")
    
    if st.session_state.signal is None:
        st.warning("⚠️ No input signal found. Please load an input signal in the **Input Signal Setup** tab first.")
    else:
        signal = st.session_state.signal
        if signal.shape != (2, 14, 612):
            st.error("❌ Expected input tensor shape of (2, 14, 612). Please upload a valid shape signal.")
        else:
            col_b, col_s = st.columns([1, 3])
            with col_b:
                run_btn = st.button("🚀 Run Channel Estimation", type="primary")
            
            # Run inference if button is clicked
            if run_btn:
                # Progress displays
                st.info("📦 Model Loaded")
                
                with st.spinner("⚡ Running Inference..."):
                    start = time.time()
                    prediction = predict(model, signal)
                    end = time.time()
                    inference_time = end - start
                
                # Save to session state
                st.session_state.prediction = prediction
                st.session_state.inference_time = inference_time
                st.success("✅ Completed Successfully")
                st.rerun()
                
            # If prediction exists in session state, display stats and downloads
            if st.session_state.prediction is not None:
                prediction = st.session_state.prediction
                inference_time = st.session_state.inference_time
                
                st.markdown("---")
                
                # Performance Card
                st.markdown(
                    f"""
                    <div class="dashboard-card" style="border-left: 5px solid #10b981;">
                        <span style="font-size: 0.95rem; color: #888; font-weight: 500;">Model Inference Speed</span>
                        <h2 style="margin: 6px 0; color: #10b981; font-weight: 700;">{inference_time:.4f} seconds</h2>
                        <span style="font-size: 0.85rem; color: #888;">Processed successfully on: <b>{device}</b></span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Output Statistics
                st.markdown("### 📊 Estimated Channel Statistics")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Minimum", f"{prediction.min():.4f}")
                with c2:
                    st.metric("Maximum", f"{prediction.max():.4f}")
                with c3:
                    st.metric("Mean", f"{prediction.mean():.4f}")
                with c4:
                    st.metric("Std", f"{prediction.std():.4f}")
                
                # Output Details
                st.markdown("### 📋 Output Details")
                st.markdown(
                    f"""
                    <div class="dashboard-card">
                        <p style="margin: 0; font-size: 0.9rem; color: #888;">Output Shape</p>
                        <h3 style="margin: 4px 0 0 0; font-weight: 700; color: #6366f1;">{prediction.shape}</h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Download Container
                st.markdown("### ⬇️ Export Estimation Results")
                st.markdown(
                    """
                    <div class="dashboard-card" style="background-color: rgba(16, 185, 129, 0.05); border: 1px dashed rgba(16, 185, 129, 0.4);">
                        <h4 style="margin-top:0; color: #10b981;">Download Estimated Tensor Data</h4>
                        <p style="font-size: 0.9rem; color: #888; margin-bottom: 15px;">
                            Export the estimated channel tensor as a standard NumPy binary (.npy) file of shape (2, 14, 612).
                        </p>
                    """,
                    unsafe_allow_html=True
                )
                
                buffer = io.BytesIO()
                np.save(buffer, prediction)
                buffer.seek(0)
                
                st.download_button(
                    label="⬇ Download Estimated Channel (.npy)",
                    data=buffer,
                    file_name="Estimated_Channel.npy",
                    mime="application/octet-stream"
                )
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.info("📈 Check the **Profile Visualizations** tab for side-by-side comparative 2D plots.")

# --------------------------------------------------
# TAB 3: 2D Channel Profile Visualizations
# --------------------------------------------------
with tab_viz:
    st.markdown("### 📊 2D Wireless Channel Profile Comparison")
    
    if st.session_state.signal is None:
        st.warning("⚠️ No input signal loaded. Please go to the **Input Signal Setup** tab first.")
    else:
        signal = st.session_state.signal
        input_real = signal[0]
        input_imag = signal[1]
        
        if st.session_state.prediction is None:
            st.info("⏳ Estimated channel response is not loaded. Showing only Input Signal channels. Run estimation to compare.")
            
            # Show input signal only
            fig, axs = plt.subplots(1, 2, figsize=(14, 4.5))
            plt.subplots_adjust(wspace=0.25)
            
            im0 = axs[0].imshow(input_real, aspect="auto", origin="lower", cmap="viridis")
            axs[0].set_title("Input Signal (Real Part)", fontsize=11, fontweight="bold", pad=8)
            axs[0].set_xlabel("Subcarriers", fontsize=9)
            axs[0].set_ylabel("OFDM Symbols", fontsize=9)
            fig.colorbar(im0, ax=axs[0])
            
            im1 = axs[1].imshow(input_imag, aspect="auto", origin="lower", cmap="plasma")
            axs[1].set_title("Input Signal (Imaginary Part)", fontsize=11, fontweight="bold", pad=8)
            axs[1].set_xlabel("Subcarriers", fontsize=9)
            axs[1].set_ylabel("OFDM Symbols", fontsize=9)
            fig.colorbar(im1, ax=axs[1])
            
            st.pyplot(fig)
            
        else:
            prediction = st.session_state.prediction
            real = prediction[0]
            imag = prediction[1]
            
            # 2x2 side-by-side comparison layout
            fig, axs = plt.subplots(2, 2, figsize=(14, 9.5))
            plt.subplots_adjust(hspace=0.35, wspace=0.25)
            
            # Row 1: Input Signal
            im00 = axs[0, 0].imshow(input_real, aspect="auto", origin="lower", cmap="viridis")
            axs[0, 0].set_title("Input Signal (Real Part)", fontsize=11, fontweight="bold", pad=8)
            axs[0, 0].set_xlabel("Subcarriers", fontsize=9)
            axs[0, 0].set_ylabel("OFDM Symbols", fontsize=9)
            fig.colorbar(im00, ax=axs[0, 0])
            
            im01 = axs[0, 1].imshow(input_imag, aspect="auto", origin="lower", cmap="plasma")
            axs[0, 1].set_title("Input Signal (Imaginary Part)", fontsize=11, fontweight="bold", pad=8)
            axs[0, 1].set_xlabel("Subcarriers", fontsize=9)
            axs[0, 1].set_ylabel("OFDM Symbols", fontsize=9)
            fig.colorbar(im01, ax=axs[0, 1])
            
            # Row 2: Estimated Channel
            im10 = axs[1, 0].imshow(real, aspect="auto", origin="lower", cmap="viridis")
            axs[1, 0].set_title("Estimated Channel (Real Part)", fontsize=11, fontweight="bold", pad=8)
            axs[1, 0].set_xlabel("Subcarriers", fontsize=9)
            axs[1, 0].set_ylabel("OFDM Symbols", fontsize=9)
            fig.colorbar(im10, ax=axs[1, 0])
            
            im11 = axs[1, 1].imshow(imag, aspect="auto", origin="lower", cmap="plasma")
            axs[1, 1].set_title("Estimated Channel (Imaginary Part)", fontsize=11, fontweight="bold", pad=8)
            axs[1, 1].set_xlabel("Subcarriers", fontsize=9)
            axs[1, 1].set_ylabel("OFDM Symbols", fontsize=9)
            fig.colorbar(im11, ax=axs[1, 1])
            
            st.pyplot(fig)

# --------------------------------------------------
# TAB 4: About RDMSNet Research details
# --------------------------------------------------
with tab_about:
    st.markdown("### ℹ️ About RDMSNet Research & Model Specs")
    
    st.markdown(
        """
        The **Residual Dense Multi-scale Squeeze-and-Excitation Network (RDMSNet)** is a deep learning architecture 
        specifically developed for **6G wireless channel estimation**. 6G communication links face complex, fast-varying 
        channels that require high-resolution channel state information (CSI) interpolation. RDMSNet addresses this 
        by processing pilot grids as image-like complex tensors and applying advanced neural blocks to reconstruct full 
        channel responses.
        """
    )
    
    st.markdown("---")
    
    col_ab1, col_ab2 = st.columns(2)
    
    with col_ab1:
        st.markdown("#### 🧱 Model Architecture Components")
        st.markdown(
            """
            * **Initial Feature Mapping Layer:** Sets up initial high-dimension representation grids via 2D convolution, batchnorm, and ReLU layers.
            * **Residual Dense Blocks (RDB):** 3 consecutive blocks that leverage continuous dense skip-connections to recover high-frequency channel variations.
            * **Residual Squeeze-and-Excitation (SE) Blocks:** 5 blocks that dynamically calibrate features and highlight active channel paths.
            * **Atrous Spatial Pyramid Pooling (ASPP):** Dilated convolutions that enable multi-scale receptivity of wide subcarrier grids.
            * **Multi-Scale Feature Fusion:** Combines multi-resolution features into a coherent dense channel response output.
            """
        )
        
    with col_ab2:
        st.markdown("#### ⚙️ Technical System Properties")
        st.markdown(
            """
            | Parameter Spec | Value / Description |
            | :--- | :--- |
            | **Model Architecture** | RDMSNet |
            | **Development Framework** | PyTorch |
            | **Total Parameters Count** | 1,093,186 |
            | **Input Tensor Shape** | (2, 14, 612) |
            | **Output Tensor Shape** | (2, 14, 612) |
            | **Dataset Samples Size** | 10,000 |
            | **Carrier Frequency** | 7 GHz |
            | **OFDM Symbols Count** | 14 |
            | **Subcarriers Count** | 612 |
            | **Supported Channels** | TDL-A, TDL-B, TDL-C, TDL-D, TDL-E |
            | **Signal-to-Noise Ratio (SNR)** | -10 dB to +30 dB |
            """
        )

# --------------------------------------------------
# Professional Page Footer
# --------------------------------------------------
st.markdown(
    """
    <hr style="margin-top: 50px; margin-bottom: 15px; border: 0; border-top: 1px solid rgba(128,128,128,0.2);">
    <div style="text-align: center; font-size: 0.85rem; color: #888; padding-bottom: 20px;">
        <p style="margin: 4px 0;">
            Developed for 6G Wireless Channel Estimation using <b>RDMSNet</b> | Powered by <b>PyTorch</b> & <b>Streamlit</b>
        </p>
        <p style="margin: 4px 0;">
            Version 2.0 • © 2026 6G Wireless Channel Estimation Dashboard
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
