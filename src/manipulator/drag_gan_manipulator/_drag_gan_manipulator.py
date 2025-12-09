from typing import Union

import numpy as np
import torch
from torch import Tensor

from ._drag_candidate import DragCandidateList
from .._manipulator import Manipulator
from ._load_draggan import load_draggan


class DragGANManipulator(Manipulator):
	"""
    A class geared towards point dragging using a DragGAN network.

    This class is heavily influenced by the Renderer class in the StyleGAN3 repo.
    """

	_generator: torch.nn.Module
	_device: torch.device
	_has_input_transform: bool
	_conditional: bool
	_noise_mode: str
	_max_iter_count: int
	_r1: int
	_r2: int
	_wn: Union[Tensor, None]
	_w_optim: Union[torch.optim.Optimizer, None]
	_lr: float

	def __init__(
			self,
			generator: Union[torch.nn.Module, str],
			device: torch.device,
			conditional: bool = True,
			noise_mode: str = "random",
			max_iter_count: int = 100,
			target_coord_system: str = "cartesian",
			r1: int = 3,
			r2: int = 12,
			lr: float = 1e-3
	):
		"""
        Initialize the Manipulator object.

        :param generator: The generator network to use or the path to its pickle file.
        :param device: The torch device to use (should be cuda).
        :param noise_mode: The noise mode to be used for generation (const, random).
        :param max_iter_count: The maximum number of iterations if the algorithm does not converge.
        :param target_coord_system: Whether the target coordinates are in Cartesian or Polar coordinates.
        :param r1: Radius of the neighborhood for motion supervision.
        :param r2: Radius of the neighborhood for point tracking.
        :param lr: The learning rate for the optimizer.
        :raises ValueError: If `noise_mode` is not supported.
        """

		if noise_mode in ["random", "const", "none"]:
			self._noise_mode = noise_mode
		else:
			raise ValueError(f"Unknown noise mode: {noise_mode}")

		self._device = device

		self._generator = (
			generator if isinstance(generator, torch.nn.Module) else load_draggan(generator)
		)
		self._generator.to(self._device)

		self._has_input_transform = hasattr(self._generator.synthesis, "input") and hasattr(
			self._generator.synthesis.input, "transform"
		)

		self._conditional = conditional

		self.target_coord_system = target_coord_system

		self._max_iter_count = max_iter_count
		self._r1 = r1
		self._r2 = r2

		self._wn = None

		self._w_optim = None
		self._lr = lr

		self._feat0_resized = None
		self._feat_refs = None

	def manipulate(
			self,
			w0: Tensor,
			candidates: DragCandidateList,
			**kwargs
	) -> Tensor:
		"""
        Generate data using style mixing or interpolation.

        This function is heavily inspired by the Renderer class of the original StyleGANv3 codebase.

		:param w0: The w of the initial image to manipulate.
        :param candidates: The candidates used for dragging operations.
        :returns: The generated image (C x H x W).
        """

		wns = torch.zeros((len(candidates), *w0.size()[1:]), device=self._device)

		if self._has_input_transform:
			m = np.eye(3)
			self._generator.synthesis.input.transform.copy_(torch.from_numpy(m))

		h, w = self._generator.img_resolution, self._generator.img_resolution
		for i, candidate in enumerate(candidates):
			self._reset(w0)

			iter_count = 0
			terminate_cond = iter_count >= self._max_iter_count

			handles = candidate.handles
			targets = candidate.targets

			while not terminate_cond:
				iter_count += 1
				grid, feat_resize = self._generate_coords(w0, handles, h, w)

				handles = self._track_points(handles, h, w, feat_resize)
				loss, is_close = self._motion_supervision(handles, targets, h, w, grid, feat_resize)

				terminate_cond = iter_count >= self._max_iter_count or is_close

				if not terminate_cond:
					self._optimize(loss)

			wns[i] = self._wn.detach().clone()

		return wns

	def _generate_coords(self,
						 w0: Tensor,
						 handles: list[tuple[int, int]],
						 h: int,
						 w: int) -> tuple[tuple[Tensor, Tensor], Tensor]:
		"""
        Generate image, the coordinate system and the resized features.
        :param w0: The w of the initial image to manipulate.
        :param handles: Handle point coordinates of the candidate.
        :param h: Height of the image.
        :param w: Width of the image.
        :return: The coordinate system and the resized features.
        """

		ws = torch.cat([self._wn[:, :6, :], w0[:, 6:, :]], dim=1)
		_, feat = self.get_image_and_features(ws)

		x = torch.linspace(0, h, h)
		y = torch.linspace(0, w, w)
		xx, yy = torch.meshgrid(x, y)
		feat_resize = torch.nn.functional.interpolate(feat[5], [h, w], mode='bilinear')

		if self._feat_refs is None:
			self._feat0_resized = torch.nn.functional.interpolate(feat[5].detach(), [h, w], mode='bilinear')
			self._feat_refs = []
			for handle in handles:
				py, px = round(handle[0]), round(handle[1])
				self._feat_refs.append(self._feat0_resized[:, :, py, px])

		return (xx, yy), feat_resize

	def _track_points(self, handles: list[tuple[int, int]], h: int, w: int, feat_resize: Tensor) -> list[
		tuple[int, int]]:
		"""
        The algorithm to track the new coordinates of the handles.
        :param handles: Handle point coordinates of the candidate.
        :param h: Height of the image.
        :param w: Width of the image.
        :param feat_resize: Resized features of the generated image.
        :return: The new coordinates of the handles.
        """

		with torch.no_grad():
			for j, handle in enumerate(handles):
				r = round(self._r2 / 512 * h)
				up = max(handle[0] - r, 0)
				down = min(handle[0] + r + 1, h)
				left = max(handle[1] - r, 0)
				right = min(handle[1] + r + 1, w)
				feat_patch = feat_resize[:, :, up:down, left:right]
				l2 = torch.linalg.norm(feat_patch - self._feat_refs[j].reshape(1, -1, 1, 1), dim=1)
				_, idx = torch.min(l2.view(1, -1), -1)
				width = right - left
				handle = [idx.item() // width + up, idx.item() % width + left]
				handles[j] = handle

			return handles

	def _motion_supervision(self,
							handles: list[tuple[int, int]],
							targets: list[tuple[int, int]],
							h: int,
							w: int,
							grid,
							feat_resize: Tensor) -> tuple[Tensor, bool]:
		"""
        The algorithm to calculate the motion suppression loss.
        :param handles: Handle point coordinates of the candidate.
        :param targets: Target point coordinates of the candidate.
        :param h: Height of the image.
        :param w: Width of the image.
        :param grid: Coordinate grid giving the x- and y-coordinates for each pixel location.
        :param feat_resize: Resized features of the generated image.
        :return: The loss and whether the handles are within acceptable bounds for termination.
        """

		loss = 0
		is_close = False
		xx, yy = grid

		for target, handle in zip(targets, handles):
			direction = torch.Tensor([target[1] - handle[1], target[0] - handle[0]])
			if torch.linalg.norm(direction) <= max(2 / 512 * h, 2):
				is_close = True
			if torch.linalg.norm(direction) > 1:
				distance = ((xx.to(self._device) - handle[0]) ** 2 + (yy.to(self._device) - handle[1]) ** 2) ** 0.5
				rel_is, rel_js = torch.where(distance < round(self._r1 / 512 * h))
				direction = direction / (torch.linalg.norm(direction) + 1e-7)
				grid_h = (rel_is + direction[1]) / (h - 1) * 2 - 1
				grid_w = (rel_js + direction[0]) / (w - 1) * 2 - 1
				grid = torch.stack([grid_w, grid_h], dim=-1).unsqueeze(0).unsqueeze(0)
				target_region = torch.nn.functional.grid_sample(feat_resize.float(), grid, align_corners=True).squeeze(2)
				loss += torch.nn.functional.l1_loss(feat_resize[:, :, rel_is, rel_js].detach(), target_region)

		return loss, is_close

	def _optimize(self, loss: Tensor) -> None:
		"""
        Optimize the loss of the dragging operation.
        :param loss: Loss of motion supervision.
        :return: None
        """

		self._w_optim.zero_grad()
		loss.backward()
		self._w_optim.step()

	@torch.no_grad()
	def _reset(self, w0: Tensor) -> None:
		"""
        Reset the wn and the optimizer.

        :param w0: The w of the initial image to manipulate.
        :return: None
        """

		self._wn = w0.detach().clone().requires_grad_(True)
		self._w_optim = torch.optim.Adam([self._wn], lr=self._lr)

	@torch.no_grad()
	def get_w(self, seed: int, class_idx: int, batch_size: int = 1) -> Tensor:
		"""
        Generate w vector(s) from a seed.

        :param seed: The seed to generate w vector(s) from.
        :param class_idx: The label of the class to generate.
        :param batch_size: How many w's should be generated.
        :returns: The w vector(s) in w+ format (B x num_ws x w_dim).
        """

		# Generate latent vectors
		z = torch.randn(size=[batch_size, self._generator.z_dim],
						generator=torch.Generator(device=self._device).manual_seed(seed),
						device=self._device)

		# Set class conditional vector, if conditional sampling is used and allowed.
		if self._generator.c_dim != 0:
			c = torch.zeros(size=[batch_size, self._generator.c_dim], device=self._device)
			c[:, class_idx] = 1 if self._conditional else 0
		else:
			c = None

		w = self._generator.mapping(z=z, c=c, truncation_psi=0.7, truncation_cutoff=None)

		return w

	def get_image_and_features(self, w: Tensor) -> tuple[Tensor, Tensor]:
		"""
        Generate an image and its features from the w vector(s).

        :param w: The w vector(s) for generation.
        :returns: The generated image (B x C x H x W) and the features.
        """

		img, feat = self._run_synthesis_net(
			self._generator.synthesis, w, return_feature=True, noise_mode=self._noise_mode, force_fp32=False
		)

		return img, feat

	@torch.no_grad()
	def get_image(self, w: Tensor) -> Tensor:
		"""
		Generate an image from the w vector(s).

		:param w: The w vector(s) for generation.
		:returns: The generated image (B x C x H x W).
		"""

		img = self._run_synthesis_net(
			self._generator.synthesis, w, return_feature=False, noise_mode=self._noise_mode, force_fp32=False
		)

		return img

	@staticmethod
	def transform_image_output(images: Tensor, normalize: bool = False, full_range: bool = False) -> Tensor:
		"""
        Transform the generated image output.

        :param images: The image to be transformed. Should be in (B x C x H x W) form.
        :param normalize: Whether to normalize the image or not.
        :param full_range: Whether to cast the image to [0,255] range or [0,1].
        :returns: The transformed image(s) in (B x C x H x W).
        """

		# Normalize color range.
		if normalize:
			images /= images.norm(float("inf"), dim=[-2, -1], keepdim=True).clip(1e-8, 1e8)

		return (images * 127.5 + 128).clamp(0, 255) if full_range else ((images + 1) / 2).clamp(0, 1)

	@staticmethod
	def _run_synthesis_net(net, *args, **kwargs) -> Union[tuple[Tensor, Tensor], Tensor]:
		"""
        Run the synthesis network.
        :param net: The network to run.
        :param args: A list of arguments to pass to the network.
        :param kwargs: A list of keyword arguments to pass to the network.
        :return: The generated image(s) and the features if return_feature is True.
        """

		return net(*args, **kwargs)
