"""
Utility functions for loading experimental stimulus data from DANDI.
"""

from pathlib import Path
from typing import Optional, List
import numpy as np
import cv2


def download_nwb_from_dandi(
    dandiset_id: str = "000951",
    nwb_index: int = 0,
    data_root: Path = Path("data"),
) -> Path:
    """
    Download an NWB file from DANDI archive.
    
    Args:
        dandiset_id: DANDI dataset ID
        nwb_index: Index of the NWB file to download (0-indexed)
        data_root: Root directory for downloaded data
        
    Returns:
        Path to the downloaded NWB file
    """
    from dandi.dandiapi import DandiAPIClient
    
    with DandiAPIClient() as client:
        ds = client.get_dandiset(dandiset_id)
        version = getattr(ds.version, "identifier", str(ds.version))
        
        assets = list(ds.get_assets())
        nwb_assets = [a for a in assets if a.path.endswith(".nwb")]
        print(f"Found {len(nwb_assets)} NWB files, version: {version}")
        
        asset = nwb_assets[nwb_index]
        out_root = data_root / dandiset_id / version
        nwb_path = out_root / asset.path
        nwb_path.parent.mkdir(parents=True, exist_ok=True)
        
        if nwb_path.exists():
            print(f"Already downloaded: {nwb_path}")
        else:
            asset.download(nwb_path)
            print(f"Downloaded: {nwb_path}")
            
    return nwb_path


def list_stimulus_keys(nwb_path: Path) -> List[str]:
    """
    List all available stimulus keys in an NWB file.
    
    Args:
        nwb_path: Path to the NWB file
        
    Returns:
        List of stimulus key names
    """
    from pynwb import NWBHDF5IO
    
    with NWBHDF5IO(str(nwb_path), "r", load_namespaces=True) as io:
        nwb = io.read()
        return list(nwb.stimulus.keys())


def get_stim_data(
    stim_key: str,
    nwb_path: Optional[Path] = None,
    dandiset_id: str = "000951",
    nwb_index: int = 0,
    max_frames: int = 100,
    target_height: int = 400,
    target_width: int = 700,
    pad_value: float = 1.0,
) -> np.ndarray:
    """
    Load stimulus frames from NWB file, preprocessed for BoxEye rendering.
    
    Downloads the NWB file if not already present, extracts the specified
    stimulus, and resizes/pads to target dimensions while preserving aspect ratio.
    
    Args:
        stim_key: Name of the stimulus in the NWB file (e.g., "nivexp_looming")
        nwb_path: Path to existing NWB file. If None, downloads from DANDI.
        dandiset_id: DANDI dataset ID (used if nwb_path is None)
        nwb_index: Index of NWB file to download (used if nwb_path is None)
        max_frames: Maximum number of frames to load
        target_height: Target height for output frames
        target_width: Target width for output frames
        pad_value: Value for padding (0.0 = black, 1.0 = white)
        
    Returns:
        numpy array of shape (1, frames, height, width) with values in [0, 1],
        ready for BoxEye rendering
    """
    from pynwb import NWBHDF5IO
    
    # Download NWB if path not provided
    if nwb_path is None:
        nwb_path = download_nwb_from_dandi(dandiset_id, nwb_index)
    
    # Load stimulus from NWB
    with NWBHDF5IO(str(nwb_path), "r", load_namespaces=True) as io:
        nwb = io.read()
        
        if stim_key not in nwb.stimulus:
            available = list(nwb.stimulus.keys())
            raise KeyError(
                f"Stimulus '{stim_key}' not found. Available: {available}"
            )
        
        series = nwb.stimulus[stim_key]
        
        if getattr(series, "data", None) is None or series.data is None:
            raise ValueError(f"Stimulus '{stim_key}' has no embedded data")
        
        # Load frames
        n_available = series.data.shape[0]
        n_frames = min(max_frames, n_available)
        stim_data = np.asarray(series.data[:n_frames])
    
    # Convert to float [0, 1]
    if stim_data.dtype == np.uint8:
        stim_data = stim_data.astype(np.float32) / 255.0
    elif stim_data.max() > 1.0:
        stim_data = (stim_data - stim_data.min()) / (stim_data.max() - stim_data.min())
    else:
        stim_data = stim_data.astype(np.float32)
    
    # Handle channel dimension - get to (frames, height, width)
    if stim_data.ndim == 4 and stim_data.shape[-1] in [1, 3]:
        stim_data = stim_data[..., 0]
    
    n_frames, h, w = stim_data.shape
    
    # Calculate scale to fit while preserving aspect ratio
    scale = min(target_height / h, target_width / w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Resize each frame preserving aspect ratio, then pad to target size
    resized_frames = []
    for i in range(n_frames):
        # Resize frame
        resized = cv2.resize(
            stim_data[i], (new_w, new_h), interpolation=cv2.INTER_LINEAR
        )
        
        # Create padded frame (letterbox/pillarbox)
        padded = np.full((target_height, target_width), pad_value, dtype=np.float32)
        pad_top = (target_height - new_h) // 2
        pad_left = (target_width - new_w) // 2
        padded[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized
        resized_frames.append(padded)
    
    # Stack and add batch dimension: (1, frames, height, width)
    return np.stack(resized_frames)[None]


def show_stim_data(
    stim_data: np.ndarray,
    fps: float = 30.0,
    max_width: int = 400,
    loop: bool = False,
) -> None:
    """
    Display stimulus data as an animated GIF inline in the notebook.
    
    Args:
        stim_data: numpy array of shape (1, frames, height, width) or (frames, height, width)
                   with values in [0, 1]
        fps: Frames per second for the animation
        max_width: Maximum width for display (scales down if larger)
        loop: If True, loop infinitely. If False, play once.
    """
    import imageio.v2 as imageio
    import tempfile
    import base64
    from IPython.display import display, HTML
    
    # Handle batch dimension
    if stim_data.ndim == 4:
        stim_data = stim_data[0]  # Remove batch dim -> (frames, height, width)
    
    n_frames, h, w = stim_data.shape
    
    # Scale down if too wide
    scale = min(1.0, max_width / w)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        frames = []
        for i in range(n_frames):
            resized = cv2.resize(stim_data[i], (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            frames.append(resized)
        stim_data = np.stack(frames)
    
    # Convert to uint8
    frames_uint8 = (np.clip(stim_data, 0, 1) * 255).astype(np.uint8)
    
    # Write to temporary GIF file
    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
        temp_path = f.name
    
    duration = int(1000 / fps)  # milliseconds per frame
    # loop=0 means infinite loop, loop=1 means play once (no repeat)
    loop_count = 0 if loop else 1
    imageio.mimsave(temp_path, frames_uint8, duration=duration, loop=loop_count)
    
    # Read and encode as base64
    with open(temp_path, "rb") as f:
        gif_data = base64.b64encode(f.read()).decode("ascii")
    
    # Display inline
    html = f'<img src="data:image/gif;base64,{gif_data}" style="max-width: 100%;"/>'
    display(HTML(html))
    
    # Clean up temp file
    Path(temp_path).unlink()

