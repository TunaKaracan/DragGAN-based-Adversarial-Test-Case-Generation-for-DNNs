import torch

from Project.src.manipulator.drag_gan_manipulator._internal.dnnlib.util import open_url
from ._internal.legacy import load_network_pkl
from ._internal.training.networks_stylegan2 import Generator as Generator2
from ._internal.training.networks_stylegan3 import Generator as Generator3


def load_draggan(pkl_path: str) -> torch.nn.Module:
    """
    Load a DragGAN network from a pickle file.

    :param pkl_path: The path of the pickle file.
    :return: The DragGAN network.
    :raises ValueError: If the model type cannot be inferred from the pickle file.
    """
    style_gan_net = load_stylegan(pkl_path)

    # Reconstruct the network since some function signatures are modified in DragGAN.
    if "stylegan2" in pkl_path:
        drag_gan_net = Generator2(*style_gan_net.init_args, **style_gan_net.init_kwargs)
    elif "stylegan3" in pkl_path:
        drag_gan_net = Generator3(*style_gan_net.init_args, **style_gan_net.init_kwargs)
    else:
        raise NameError(f"Cannot infer model type from pkl name {pkl_path}!")

    drag_gan_net.load_state_dict(style_gan_net.state_dict())
    return drag_gan_net


def load_stylegan(pkl_path: str) -> torch.nn.Module:
    """
    Load a StyleGAN network from a pickle file.

    :param pkl_path: The path of the pickle file.
    :returns: The StyleGAN network.
    :raises FileNotFoundError: If the pickle file cannot be found.
    """

    print(f"Loading {pkl_path}...", flush=True)
    try:
        with open_url(pkl_path) as f:
            return load_network_pkl(f)["G_ema"]
    except FileNotFoundError:
        raise FileNotFoundError(f"File {pkl_path} not found.")
