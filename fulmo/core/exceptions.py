# flake8: noqa


class PipelineCriterionBuildError(Exception):
    """Raises during `_build_criterion`."""

    pass


class PipelineMetricBuildError(Exception):
    """Raises during `_build_metrics`."""

    pass


class PipelineModelBuildError(Exception):
    """Raises during `_build_model`."""

    pass


class PipelineOptimizerBuildError(Exception):
    """Raises during `configure_optimizers`."""

    pass


class PipelineStepError(Exception):
    """Raises during computes losses/metrics."""

    pass


__all__ = [
    "PipelineCriterionBuildError",
    "PipelineMetricBuildError",
    "PipelineModelBuildError",
    "PipelineOptimizerBuildError",
    "PipelineStepError",
]
