# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2021-11-23

### Add

- **registry**: add scheduler registry 
- **registry**: add model registry

## [0.3.2] - 2021-09-02

### Refactor

- **checker**: add mypy stubs
- **logging**: changing parameters of described objects in the config for debug stage
- **wrapper**: move the reduction function from the `BaseModule' to the loss wrapper
- **callback**: rename `apply_after_epoch` to `apply_on_epoch`

### Feat

- **checker**: add static check for the whole project in pre-commit-config
- **core**: add an unfreezing function for batch normalization layers
- **core**: add support for two-step optimizers
- **core**: check the config before starting the pipeline

### Fix

- **model**: add `pool_parameters`
- **model**: turn on bias in the last layer if batch normalization is off

## [0.3.1] - 2021-08-17

### Fix

- **chore**: fix path to file with version
- **tests**: fix relative imports
- **ema**: fix case when `EmaCallback` using with `FreezeUnfreezeBackboneCallback` at the same time
- **callback**: Fix race condition for instances of `BaseMixCallback`

### Feat

- **ci**: bump version from commit
- **dataset**: add `MultiDomainCsvDataset` dataset
- **callback**: add `ScheduledDropoutCallback` callback
- **ci**: remove `bump2version` tool
- **ci**: add `commitizen` tool

### Refactor

- **model**: rename field encoder to backbone
- **callback**: replace `FreezeUnfreezeCallback` with `FreezeUnfreezeBackboneCallback` from `pytorch_lightning.callbacks`
- **core**: The configuration is no longer changed during module initialization

### Add

- **core**: Different `BaseDataModule` for each stage
- **reader**: `NpyReader`
- **ci**: First release on PyPI.

[Unreleased]: https://github.com/jexio/fulmo/compare/v1.0.0...develop
[1.0.0]: https://github.com/jexio/fulmo/compare/v0.3.2...v1.0.0
[0.3.2]: https://github.com/jexio/fulmo/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/jexio/fulmo/compare/releases/tag/v0.3.1
