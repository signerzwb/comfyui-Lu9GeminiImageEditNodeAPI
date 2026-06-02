import io


def ensure_pil():
    from PIL import Image

    return Image


def ensure_numpy():
    import numpy as np

    return np


def ensure_torch():
    import torch

    return torch


def tensor_to_png_bytes(image_tensor) -> bytes:
    Image = ensure_pil()
    np = ensure_numpy()
    torch = ensure_torch()

    if image_tensor is None:
        raise ValueError("Image tensor is None")

    if not isinstance(image_tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(image_tensor)!r}")

    tensor = image_tensor.detach().cpu()

    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise ValueError(f"Expected image tensor with 3 dims after batch strip, got {tensor.ndim}")

    if tensor.shape[-1] in (3, 4):
        array = tensor.numpy()
    elif tensor.shape[0] in (3, 4):
        array = tensor.permute(1, 2, 0).numpy()
    else:
        raise ValueError(f"Unsupported tensor shape for image conversion: {tuple(tensor.shape)}")

    array = np.clip(array, 0.0, 1.0)
    array = (array * 255.0).round().astype("uint8")
    mode = "RGBA" if array.shape[-1] == 4 else "RGB"
    image = Image.fromarray(array, mode=mode)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
