import logging
import os
from itertools import product
from time import time
from typing import Any, Optional
import json

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import torch
from torch import Tensor
import plotly.graph_objects as go

from src import SMOO, TEarlyTermCallable
from src.manipulator.drag_gan_manipulator import (
	DragCandidate,
	DragCandidateList,
	DragGANManipulator
)
from src.objectives import CriterionCollection
from src.optimizer import Optimizer
from src.sut import SUT

from Project.defaults.project._experiment_config import ExperimentConfig


class ProjectTester(SMOO):
	"""A tester class for DNN using latent space manipulation in generative models (DragGAN)."""

	_config: ExperimentConfig

	_manipulator: DragGANManipulator

	def __init__(
			self,
			*,
			sut: SUT,
			manipulator: DragGANManipulator,
			optimizer: Optimizer,
			objectives: CriterionCollection,
			config: ExperimentConfig,
			early_termination: Optional[TEarlyTermCallable] = None
	):
		super().__init__(
			sut=sut,
			manipulator=manipulator,
			optimizer=optimizer,
			objectives=objectives,
			restrict_classes=None,
			use_wandb=False
		)

		self._config = config
		self._early_termination = early_termination or (lambda _: (False, None))
		self._term_early: bool = False

	def test(self) -> None:
		script_dir = os.path.dirname(os.path.abspath(__file__))
		spc, c, seeds = self._config.samples_per_class, self._config.classes, self._config.seeds
		plot_cols_names = np.array(
			[[f"{c_n}_max", f"{c_n}_90", f"{c_n}_avg", f"{c_n}_10", f"{c_n}_min"] for c_n in self._objectives.names]
		)

		logging.info(
			f"Start tests. Number of classes: {len(c)}, iterations per class: {spc}, total iterations: {len(c) * spc}\n"
		)

		img_resolution = self._manipulator._generator.img_resolution

		for class_idx, sample_id in product(c, range(spc)):
			logging.info(f"Test class {class_idx}, sample idx {sample_id}.")
			start_time = time()  # Stores the start time of the current experiment.
			# We should be generating a single w every time
			w0_tensor, w0_image, w0_y, w0_trial, chosen_seeds = self._generate_seeds(1, class_idx, seed=seeds[sample_id])
			seeds[sample_id] = chosen_seeds[0]

			self._objectives.precondition_all({
				"logits": w0_y,
			})

			all_gen_data: list[dict[str, Any]] = []

			logging.info(f"Running Search-Algorithm for {self._config.generations} generation(s).")
			for gen in range(self._config.generations):
				gen_start = time()
				logging.info(f"Generation {gen + 1} started.")

				x_current = (self._optimizer.get_x_current() * img_resolution).astype(np.int32)
				handles = x_current[:, :, 0, :]
				targets = x_current[:, :, 1, :]

				targets = self.transform_coordinates(handles, targets, img_resolution)

				drag_list = DragCandidateList(
					*[DragCandidate(handle, target) for handle, target in zip(handles, targets)]
				)

				wn_tensor = self._manipulator.manipulate(w0_tensor, drag_list)
				wn_image = self._manipulator.transform_image_output(self._manipulator.get_image(wn_tensor))
				wn_image = self._assure_rgb(wn_image)
				wn_y = self._process(wn_image)

				self._objectives.evaluate_all({
					"images": torch.vstack((w0_image, wn_image)),
					"logits": wn_y,
					"label_targets": [w0_y.argmax(dim=1).item(), -1],
					"handles": torch.from_numpy(handles),
					"targets": torch.from_numpy(targets),
					"img_resolution": img_resolution,
					"batch_dim": 0
				})

				fitness = tuple(np.asarray(f) for f in self._objectives.results.values())
				self._optimizer.assign_fitness(
					fitness,
					[wn_image[i] for i in range(wn_image.shape[0])],
					wn_y.tolist())
				logging.info(f"Generation {gen + 1} done in {time() - gen_start:.2f}s.")

				gen_data = {
					"generation": gen + 1,
				}
				for i, names in enumerate(plot_cols_names):
					gen_data |= {names[0]: np.max(fitness[i])}
					gen_data |= {names[1]: np.percentile(fitness[i], 90)}
					gen_data |= {names[2]: np.mean(fitness[i])}
					gen_data |= {names[3]: np.percentile(fitness[i], 10)}
					gen_data |= {names[4]: np.min(fitness[i])}
				gen_data |= self._objectives.results
				all_gen_data.append(gen_data)

				early_term, term_cond = self._early_termination(self._objectives.results)
				self._term_early = early_term
				if early_term and term_cond:
					logging.info(f"Early termination condition met by: {term_cond.sum()} individuals")

				if self._term_early:
					logging.info(
						f"Early termination triggered at generation {gen + 1} by {np.sum(term_cond)} individuals."
					)
					break

				if gen != self._config.generations - 1:
					self._optimizer.update()

			log_dir = os.path.join(
				script_dir, f"runs/{self._config.save_as}_class_{class_idx}_{time()}"
			)
			os.makedirs(log_dir, exist_ok=True)

			df = pd.DataFrame(all_gen_data)
			df.to_csv(log_dir + "/data.csv", index=False)

			fig = go.Figure()

			for i, obj in enumerate(plot_cols_names):
				for name in obj:
					fig.add_trace(
						go.Scatter(
							x=df["generation"].to_numpy(),
							y=df[name].to_numpy(),
							mode="lines+markers",
							name=name,
							legendgroup=i,
							legendgrouptitle_text=self._objectives.names[i],
							hovertemplate=f"{name}<br>Generation: %{{x}}<br>Value: %{{y}}<extra></extra>"
						)
					)

			fig.update_layout(
				title="Fitness over Generations",
				xaxis_title="Generation",
				yaxis_title="Value",
				legend=dict(groupclick="toggleitem", indentation=20),
				legend_title="Legend",
				template="plotly_white",
				width=1500,
				height=900
			)
			fig.write_html(log_dir + "/fitness.html", include_plotlyjs="cdn", full_html=True)

			# Compile comprehensive stats
			stats = {
				"runtime": time() - start_time,
				"num_generations": self._config.generations,
				"seed": seeds[sample_id],
				"w0_predictions": w0_y.cpu().squeeze().tolist(),
				"best_stats": {}
			}

			for i, bc in enumerate(self._optimizer.best_candidates):
				solution = (bc.solution * img_resolution).astype(np.int32)
				handles = solution.reshape(-1, 4)[np.newaxis, :, :2]
				targets = solution.reshape(-1, 4)[np.newaxis, :, 2:]

				targets = self.transform_coordinates(handles, targets, img_resolution)

				self._save_tensor_as_image(bc.data[0], log_dir + f"/best_{i}.png")
				self._save_tensor_as_image(
					w0_image,
					log_dir + f"/og_{i}.png",
					handles=handles,
					targets=targets,
					img_resolution=img_resolution
				)
				stats["best_stats"][f"best_{i}"] = {}
				stats["best_stats"][f"best_{i}"]["y_hat"] = bc.data[1]
				stats["best_stats"][f"best_{i}"]["fitness"] = list(bc.fitness)
				stats["best_stats"][f"best_{i}"]["solution_cont"] = bc.solution.tolist()
				stats["best_stats"][f"best_{i}"]["solution_disc"] = solution.tolist()

				threshold = 0.5 if self._config.apply_activation == "sigmoid" else 0

				w0_bin = [1 if x > threshold else 0 for x in w0_y.squeeze()]
				wn_bin = [1 if x > threshold else 0 for x in bc.data[1]]

				did_change = [1 if a != b else 0 for a, b in zip(w0_bin, wn_bin)]

				stats["best_stats"][f"best_{i}"]["did_change"] = did_change

			# Save origin and target images
			self._save_tensor_as_image(w0_image, log_dir + f"/og_{class_idx}_{seeds[sample_id]}.png")

			# Save stats as JSON
			with open(f"{log_dir}/stats.json", "w") as f:
				json.dump(stats, f, indent=4)

			logging.info(
				f"Best candidate(s) have a fitness of: {', '.join([str(c.fitness) for c in self._optimizer.best_candidates])}"
			)

			self._optimizer.update_problem()  # Reset the learner to have clean slate in next iteration.
			logging.info("Reset learner!")

	def _generate_seeds(
			self,
			amount: int,
			label: int,
			seed: Optional[int] = None,
			force_unique: bool = True
	) -> tuple[Tensor, Tensor, Tensor, int, list[int]]:

		ws: list[Tensor] = []
		imgs: list[Tensor] = []
		y_hats: list[Tensor] = []

		chosen_seeds = []

		logging.info(f"Generating seed(s) for class: {label}.")

		# For logging purposes to see how many samples we need to find valid seed.
		trials = 0
		seed = seed or self._get_time_seed()
		while len(ws) < amount:
			# We generate w latent vector.
			w = self._manipulator.get_w(seed + trials, label)  # Adding trial.
			# We generate and transform the image to RGB if it is in Grayscale.
			img = self._manipulator.transform_image_output(self._manipulator.get_image(w))
			img = self._assure_rgb(img)
			y_hat = self._process(img)

			reject_seed = force_unique and seed + trials in chosen_seeds

			# We are only interested in a candidate if the prediction matches the label.
			# If the generator is unconditional then the initial prediction is assumed to be correct.
			if ((y_hat.argmax().item() == label) or (label == -1)) and not reject_seed:
				ws.append(w)
				imgs.append(img)
				y_hats.append(y_hat)
				chosen_seeds.append(seed + trials)

			trials += 1

		logging.info(f"Found {amount} valid seed(s) after: {trials} iteration(s).")

		# Convert lists to batched tensors
		ws_tensor = torch.cat(ws)
		images_tensor = torch.cat(imgs)
		y_hats_tensor = torch.cat(y_hats)

		return ws_tensor, images_tensor, y_hats_tensor, trials, chosen_seeds

	def transform_coordinates(self, handles: NDArray, targets: NDArray, img_resolution: int) -> NDArray:
		if self._config.target_coord_system == "polar":
			angles = targets[:, :, 0] / img_resolution * 2 * np.pi
			mags = targets[:, :, 1]

			targets[:, :, 0] = handles[:, :, 0] + mags * np.sin(angles)
			targets[:, :, 1] = handles[:, :, 1] + mags * np.cos(angles)

			targets = np.clip(targets, 0, img_resolution).astype(np.int32)

		return  targets
