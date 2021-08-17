<div align="center">

[![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/jexio/fulmo?logo=github)](https://github.com/jexio/fulmo/releases)
[![Python Version](https://img.shields.io/badge/python-3.8%20%7C%203.9-blue?logo=python)](https://www.python.org/)
[![Tests](https://github.com/jexio/fulmo/workflows/tests/badge.svg)](https://github.com/jexio/fulmo/actions?workflow=tests)
[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg?style=flat-square)](https://conventionalcommits.org)

[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

</div>

# fulmo

Template to start your deep learning project based on `PyTorchLightning` for rapid prototyping.

**Contents**
- [fulmo](#fulmo)
  - [Why Lightning + Hydra + Albumentations?](#why-lightning--hydra--albumentations)
  - [Features](#features)
  - [Project Structure](#project-structure)
  - [Workflow](#workflow)
  - [Experiment Tracking](#experiment-tracking)
  - [Quick start](#quickstart)
  - [Todo](#todo)
  - [Credits](#credits)
<br>
    
## Why Lightning + Hydra + Albumentations?
- [PyTorch Lightning][PyTorchLightning/pytorch-lightning] provides great abstractions for well structured ML code and advanced features like checkpointing, gradient accumulation, distributed training, etc.
- [Hydra][facebookresearch/hydra] provides convenient way to manage experiment configurations and advanced features like overriding any config parameter from command line, scheduling execution of many runs, etc.
- [Albumentations][albumentations-team/albumentations] (**Optional**) provides many image augmentation. Albumentations supports all common computer vision tasks such as classification, semantic segmentation, instance segmentation, object detection, and pose estimation. 
<br>

## Features

Pipelines based on hydra-core configs and PytorchLightning modules
- Predefined folder structure. Modularity: all abstractions are split into different submodule
- Rapid Experimentation. Thanks to automating pipeline with config files and hydra command line superpowers
- Little Boilerplate. So pipeline can be easily modified
- Main Configuration. Main config file specifies default training configuration
- Experiment Configurations. Stored in a separate folder, they can be composed out of smaller configs, override chosen parameters or define everything from scratch
- Experiment Tracking. Many logging frameworks can be easily integrated
- Logs. All logs (checkpoints, data from loggers, chosen hparams, etc.) are stored in a convenient folder structure imposed by Hydra 
- Automates PyTorch Lightning training pipeline with little boilerplate, so it can be easily modified
- Augmentations with [albumentations][albumentations-team/albumentations] described in a yaml config.
- Support of [timm models][rwightman/pytorch-image-models], [pytorch-optimizer][jettify/pytorch-optimizer] and [TorchMetrics][PyTorchLightning/pytorch-metrics]
- Exponential Moving Average for a more stable training, and Stochastic Moving Average for a better generalization and just overall performance.

<br>

## Project structure
The directory structure of new project looks like this: 
```
├── src
│   ├── fulmo
│   │   ├── callbacks               <- PyTorch Lightning callbacks
│   │   ├── core                    <- PyTorch Lightning models
│   │   ├── datasets                <- PyTorch datasets
│   │   ├── losses                  <- PyTorch losses
│   │   ├── metrics                 <- PyTorch metrics  
│   │   ├── models                  <- PyTorch model architectures
│   │   ├── optimizers              <- PyTorch optimizers
│   │   ├── readers                 <- Data readers
│   │   ├── samples                 <- PyTorch samplers
│   │   ├── schedulers              <- PyTorch schedulers
│   │   └── utils
├── tests
│   ├── test_fulmo                  <- Tests
│
├── .bumpversion.cfg
├── .darglint
├── .gitignore
├── .pre-commit-config.yaml <- Configuration of hooks for automatic code formatting
├── CHANGELOG.md
├── mypy.ini
├── noxfile.py
├── poetry.lock             <- File for installing python dependencies
├── pyproject.toml          <- File for installing python dependencies
├── README.md
└── tasks.py
```

<br>

## Workflow
1. Write your PyTorch model
2. Write your PyTorch Lightning datamodule
3. Write your experiment config, containing paths to your model and datamodule
4. Run training with chosen experiment config:<br>
```bash
python train.py +experiment=experiment_name
```
<br>

## Experiment Tracking
PyTorch Lightning provides built in loggers for Weights&Biases, Neptune, Comet, MLFlow, Tensorboard and CSV. To use one of them, simply add its config to [configs/logger](configs/logger) and run:
 ```yaml
python train.py logger=logger_name
```
<br>

## Quickstart

<details>
<summary>First, install dependencies</summary>

```yaml
pip install fulmo | poetry add fulmo
```

</details>

<details>
<summary>Second, create your project</summary>

See [examples](https://github.com/jexio/g2net/tree/master/configs) folder.

</details>

<details>
<summary>Next, you can train model with default configuration without logging</summary>

```yaml
python train.py
```

</details>

<details>
<summary>Or you can train model with chosen experiment config</summary>

```yaml
python train.py +experiment=experiment_name
```

</details>

<details>
<summary>Resume from a checkpoint</summary>

```yaml
# checkpoint can be either path or URL
# path should be either absolute or prefixed with `${work_dir}/`
# use quotes '' around argument or otherwise $ symbol breaks it
python train.py '+trainer.resume_from_checkpoint=${work_dir}/logs/runs/2021-06-23/16-50-49/checkpoints/last.ckpt'
```

</details>

<br>

## TODO
- [Data version control][dvc]
- Metric learning pipeline
- Integrate [Cross-Batch Memory for Embedding Learning (XBM)][msight-tech/research-xbm]
- Image augmentation policies

<br>

## Credits
* This package was created with [Cookiecutter][cookiecutter] and the [fedejaure/cookiecutter-modern-pypackage][cookiecutter-modern-pypackage] project template.
* [hobogalaxy/lightning-hydra-template][hobogalaxy/lightning-hydra-template]
* [Erlemar/pytorch_tempest][Erlemar/pytorch_tempest]
* [bonlime/pytorch-tools][bonlime/pytorch-tools]


[cookiecutter]: https://github.com/cookiecutter/cookiecutter
[cookiecutter-modern-pypackage]: https://github.com/fedejaure/cookiecutter-modern-pypackage
[PyTorchLightning/pytorch-lightning]: https://github.com/PyTorchLightning/pytorch-lightning
[PyTorchLightning/pytorch-metrics]: https://github.com/PytorchLightning/metrics
[hobogalaxy/lightning-hydra-template]: https://github.com/hobogalaxy/lightning-hydra-template
[albumentations-team/albumentations]: https://github.com/albumentations-team/albumentations
[facebookresearch/hydra]: https://github.com/facebookresearch/hydra
[rwightman/pytorch-image-models]: https://github.com/rwightman/pytorch-image-models
[jettify/pytorch-optimizer]: https://github.com/jettify/pytorch-optimizer
[bonlime/pytorch-tools]: https://github.com/bonlime/pytorch-tools
[Erlemar/pytorch_tempest]: https://github.com/Erlemar/pytorch_tempest
[msight-tech/research-xbm]: https://github.com/msight-tech/research-xbm
[mlflow]: https://mlflow.org/
[dvc]: https://dvc.org/
[ClearML]: https://clear.ml/
[commitizen-tools/commitizen]: https://github.com/commitizen-tools/commitizen
