import logging

import torch
from torchvision import models, transforms

from Project.defaults.project._project_tester import ProjectTester

from src.manipulator.drag_gan_manipulator._drag_gan_manipulator import DragGANManipulator

from src.objectives._criterion_collection import CriterionCollection
from defaults.objective_configs import (DYNAMIC_ADVERSARIAL_TESTING,
                                        MULTI_ATTRIBUTE_ADVERSARIAL_TESTING,
                                        MULTI_ATTRIBUTE_ADVERSARIAL_TESTING2)

from src.optimizer._pymoo_optimizer import PymooOptimizer
from defaults.optimizer_configs import PYMOO_AGE_MOEA_DEFAULT_PARAMS

from src.sut._classifier_sut import ClassifierSUT

from defaults.project._experiment_config import ExperimentConfig

logging.basicConfig(level=logging.INFO)

GENERATOR_PATH = "/home/tuna/PycharmProjects/Thesis/Project/models/generators/stylegan2-celebahq-256x256.pkl"
PREDICTOR_PATH = "/home/tuna/PycharmProjects/Thesis/Project/models/predictors/resnet50-celebahq-256x256_all.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

exp_config = ExperimentConfig(samples_per_class=25,
							  generations=10,
							  target_coord_system="cartesian",
							  apply_activation="none",
							  classes=[-1],
							  seeds=[],
							  save_as="multi_att")
exp_config.save_as = f"{exp_config.save_as}_{exp_config.target_coord_system}_{exp_config.apply_activation}"

manipulator = DragGANManipulator(GENERATOR_PATH,
								 DEVICE,
								 noise_mode="const",
								 conditional=False,
								 max_iter_count=200,
								 target_coord_system=exp_config.target_coord_system)

objective_config = MULTI_ATTRIBUTE_ADVERSARIAL_TESTING
objectives = CriterionCollection(*objective_config)

optimizer_config = PYMOO_AGE_MOEA_DEFAULT_PARAMS
optimizer = PymooOptimizer(**optimizer_config, num_objectives=len(objective_config), target_coord_system=exp_config.target_coord_system)

predictor = models.resnet50(pretrained=True)
state_dict = torch.load(PREDICTOR_PATH)
state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
num_features = predictor.fc.in_features
predictor.fc = torch.nn.Linear(num_features, 40)
predictor.load_state_dict(state_dict)

transformer = torch.nn.Sequential(
    transforms.Resize((224, 224)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
)

sut = ClassifierSUT(predictor, apply_activation=exp_config.apply_activation, transformer=transformer, device=DEVICE)

tester = ProjectTester(sut=sut, manipulator=manipulator, optimizer=optimizer, objectives=objectives, config=exp_config)
tester.test()
