import os
import numpy as np
import torch
from deployment_model import RDMSNet

def get_device():
    """
    Automatically detects and returns the best available device (CUDA, MPS, or CPU).
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def load_model(checkpoint_path=None, device=None):
    """
    Loads the RDMSNet model. If checkpoint_path is provided, loads the state dict
    from the checkpoint file. Sets the model to eval mode.
    
    Parameters:
        checkpoint_path (str, optional): Path to the model checkpoint.
        device (torch.device or str, optional): Device to load the model on.
                                                If None, auto-detects.
    
    Returns:
        RDMSNet: The instantiated and loaded model in eval mode.
    """
    if device is None:
        device = get_device()
    else:
        device = torch.device(device)
        
    model = RDMSNet()
    
    if checkpoint_path is not None:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("model_state_dict", checkpoint)
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)
        
    model.to(device)
    model.eval()
    return model

def predict(model, input_data, device=None):
    """
    Runs model inference on the provided input data.
    
    Parameters:
        model (nn.Module): The loaded RDMSNet model.
        input_data (numpy.ndarray or torch.Tensor): Input data.
        device (torch.device or str, optional): Device to run inference on.
                                                If None, uses model's device.
                                                
    Returns:
        numpy.ndarray: NumPy array containing the model's prediction.
    """
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)
        
    # Check if input is a PyTorch tensor or a NumPy array
    is_tensor = isinstance(input_data, torch.Tensor)
    if is_tensor:
        x = input_data.clone().detach().to(device)
    else:
        x = torch.from_numpy(np.asarray(input_data, dtype=np.float32)).to(device)
        
    original_ndim = x.ndim
    
    # Automatically add batch dimension if needed
    if original_ndim == 3:
        x = x.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)
    elif original_ndim != 4:
        raise ValueError(f"Expected 3D input of shape (2, 14, 612) or 4D input of shape (1, 2, 14, 612), got ndim={original_ndim} with shape {input_data.shape}")
        
    # Standard check for shape (..., 14, 612) and transposition to (..., 612, 14)
    # The RDMSNet model was trained with spatial dimensions (612, 14)
        
    model.eval()
    with torch.no_grad():
        out = model(x)
        
    # Convert output back to NumPy array on CPU
    out_np = out.cpu().numpy()
    
        
    # Remove batch dimension if it was originally added
    if original_ndim == 3:
        out_np = np.squeeze(out_np, axis=0)
        
    return out_np
