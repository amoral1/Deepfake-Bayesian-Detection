import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import timm
import numpy as np
import os
import random

# ===========================
# 1. Basic setting
# ===========================
st.set_page_config(page_title="Deepfake Probabilistic Detector", layout="wide")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===========================
# 2. Models
# ===========================

# --- A. MC Dropout ---
class MCDropoutMLP(nn.Module):
    def __init__(self, d_in=2048, hidden=256, p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x)

# --- B. VI  ---
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_std=0.5):
        super().__init__()
        self.w_mu  = nn.Parameter(torch.zeros(out_features, in_features))
        self.w_rho = nn.Parameter(torch.full((out_features, in_features), -3.0))
        self.b_mu  = nn.Parameter(torch.zeros(out_features))
        self.b_rho = nn.Parameter(torch.full((out_features,), -3.0))
        self.prior_std = prior_std

    def forward(self, x):
        w_sigma = F.softplus(self.w_rho)
        b_sigma = F.softplus(self.b_rho)
        w = self.w_mu + w_sigma * torch.randn_like(w_sigma)
        b = self.b_mu + b_sigma * torch.randn_like(b_sigma)
        return F.linear(x, w, b)

class VIModel(nn.Module):
    def __init__(self, in_dim=2048):
        super().__init__()
        self.layer = BayesianLinear(in_dim, 1)

    def forward(self, x):
        return self.layer(x)

# ===========================
# 3. core functions
# ===========================

@st.cache_resource
def load_feature_extractor():
    """Loading Xception Feature Extraction"""
    try:
        # FIX: use 'legacy_xception' to avoid deprecation warning
        model = timm.create_model("legacy_xception", pretrained=True, num_classes=0)
        model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Failed to load feature extractor: {e}")
        return None

def safe_load_checkpoint(filepath):
    try:
        #  PyTorch 2.6+  weights_only=False
        return torch.load(filepath, map_location=DEVICE, weights_only=False)
    except TypeError:
        #  If old PyTorch (does not support weights_only)
        return torch.load(filepath, map_location=DEVICE)
    except Exception as e:
        raise e

def load_head_model(filepath, architecture_type):
    if architecture_type == "mc_dropout":
        model = MCDropoutMLP(d_in=2048, hidden=256, p=0.3)
    elif architecture_type == "vi":
        model = VIModel(in_dim=2048)
    else:
        return None, None, None

    if not os.path.exists(filepath):
        return None, None, None

    try:
        checkpoint = safe_load_checkpoint(filepath)
        
        mu = None
        sigma = None
        if isinstance(checkpoint, dict):
            if "mu" in checkpoint:
                mu = checkpoint["mu"]
            if "sigma" in checkpoint:
                sigma = checkpoint["sigma"]

        # state_dict
        state_dict = None
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                # filter non-state_dict（ mu, sigma, model_type ...）
                known_meta_keys = {"mu", "sigma", "model_type"}
                filtered = {k: v for k, v in checkpoint.items() if k not in known_meta_keys}
                
                if filtered and all(isinstance(v, torch.Tensor) for v in filtered.values()):
                    state_dict = filtered
                else:
                    state_dict = checkpoint
        else:
            state_dict = checkpoint

        # === Key fix: Address the issue of mismatched key prefixes in the VI model. ===
        if architecture_type == "vi":
            # check layer prefix
            has_layer_prefix = any(k.startswith("layer.") for k in state_dict.keys())
            if not has_layer_prefix:
                new_state_dict = {}
                for k, v in state_dict.items():
                    # add layer prefix
                    new_key = f"layer.{k}" if not k.startswith("layer.") else k
                    new_state_dict[new_key] = v
                state_dict = new_state_dict
        # ===============================================

        model.load_state_dict(state_dict, strict=False) # strict=False
        model.to(DEVICE)
        return model, mu, sigma
    except Exception as e:
        st.error(f"⚠️ Error loading {filepath}: {str(e)}")
        return None, None, None

def process_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')

    transform = T.Compose([
        T.Resize(342),
        T.CenterCrop(299),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0).to(DEVICE)

def predict_uncertainty(feature_extractor, head_model, img_tensor, model_type,
                        mu=None, sigma=None, n_samples=50):
    
    # 1. feature extraction
    with torch.no_grad():
        features = feature_extractor(img_tensor)

    # 2. normalization
    if mu is not None and sigma is not None:
        mu_dev = mu.to(DEVICE)
        sigma_dev = sigma.to(DEVICE)
        features = (features - mu_dev) / sigma_dev

    # 3. prepare models
    if model_type == "mc_dropout":
        head_model.train() # MC Dropout only
    else:
        head_model.eval()

    probs = []
    with torch.no_grad():
        for _ in range(n_samples):
            logits = head_model(features)
            p = torch.sigmoid(logits).item()
            probs.append(p)

    probs = np.array(probs)
    return probs.mean(), probs.std()

# ===========================
# 4. Streamlit
# ===========================

# --- sidebar ---
st.sidebar.header("🛠️ Model Configuration")
st.sidebar.write("Select models to run:")

# models
model_options = {
    "MC Dropout": {"file": "checkpoints/mc_dropout.pt", "type": "mc_dropout"},
    "Linear": {"file": "checkpoints/bayesian_linear.pt", "type": "vi"}, # 假设这也是 VI 结构
    "Variational Inference": {"file": "checkpoints/variational_inference.pt", "type": "vi"}
}

available_defaults = [name for name, cfg in model_options.items() if os.path.exists(cfg["file"])]
if not available_defaults:
    available_defaults = ["MC Dropout"] # Fallback

selected_models = st.sidebar.multiselect(
    "Active Models", 
    options=list(model_options.keys()),
    default=available_defaults
)

# --- main ---
st.title("Deepfake Detection based on Bayesian Models")
st.markdown("Analyze images using **probability score** to detect AI-generated content.")

# initialize Session State
if 'selected_image' not in st.session_state:
    st.session_state.selected_image = None

# --- sample pics ---
with st.expander("📂 Try a Sample Image", expanded=True):
    sample_dir = "sample_pics"
    
    if os.path.exists(sample_dir):
        valid_ext = ('.png', '.jpg', '.jpeg', '.webp')
        all_files = [f for f in os.listdir(sample_dir) if f.lower().endswith(valid_ext)]
        
        if all_files:
            # 4 at most
            if 'random_samples' not in st.session_state:
                st.session_state.random_samples = random.sample(all_files, min(len(all_files), 4))
            
            cols = st.columns(len(st.session_state.random_samples))
            for idx, file_name in enumerate(st.session_state.random_samples):
                file_path = os.path.join(sample_dir, file_name)
                with cols[idx]:
                    try:
                        img = Image.open(file_path).convert('RGB')
                        st.image(img, width='stretch')
                        if st.button(f"Analyze Sample {idx+1}", key=f"btn_{idx}"):
                            st.session_state.selected_image = img
                    except Exception as e:
                        st.error(f"Bad image: {file_name}")
        else:
            st.warning(f"No valid images found in '{sample_dir}'.")
    else:
        st.info(f"Note: Create a folder named '{sample_dir}' and verify your images are inside.")

st.divider()
uploaded_file = st.file_uploader("Or upload your own image", type=["jpg", "png", "jpeg", "webp"])

if uploaded_file:
    try:
        st.session_state.selected_image = Image.open(uploaded_file).convert('RGB')
    except Exception as e:
        st.error(f"Error reading image: {e}")

if st.session_state.selected_image is not None:
    col_img, col_res = st.columns([1, 2])
    
    with col_img:
        st.image(st.session_state.selected_image, caption="Target Image", width='stretch')
    
    with col_res:
        st.subheader("Inference Results")
        
        if not selected_models:
            st.warning("Please select at least one model from the sidebar to start analysis.")
        else:
            with st.spinner("Loading Feature Extractor..."):
                feature_extractor = load_feature_extractor()
            
            if feature_extractor:
                img_tensor = process_image(st.session_state.selected_image)

                for model_name in selected_models:
                    config = model_options[model_name]
                    
                    head, mu, sigma = load_head_model(config["file"], config["type"])
                    
                    if head:
                        mean_p, std_dev = predict_uncertainty(
                            feature_extractor, head, img_tensor, config["type"],
                            mu=mu, sigma=sigma
                        )
                        
                        with st.container():
                            st.markdown(f"### {model_name}")
                            m_col1, m_col2, m_col3 = st.columns(3)
                            
                            m_col1.metric("Fake Probability Score", f"{mean_p:.2%}")
                            m_col2.metric("Uncertainty (Std)", f"{std_dev:.4f}")
                            
                            
                            st.progress(float(mean_p))
                            
                            if std_dev > 0.1:
                                st.caption("High uncertainty: The model is not sure about this image.")
                            else:
                                st.caption("Low uncertainty: The model is confident.")
                            
                            st.divider()
                    else:
                        if os.path.exists(config["file"]):
                             st.error(f"Failed to load **{model_name}**.")
                        else:
                             st.warning(f"File **{config['file']}** not found. Please upload it.")
