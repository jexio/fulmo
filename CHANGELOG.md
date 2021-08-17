# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.1.0] - 2021-08-17

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

[Unreleased]: https://github.com/jexio/fulmo/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/jexio/fulmo/compare/releases/tag/v0.1.0
