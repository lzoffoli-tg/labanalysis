"""Test suite for labanalysis.pipelines module."""

import pytest
from labanalysis import pipelines


def test_pipelines_module_importable():
    """Test that pipelines module imports successfully."""
    assert pipelines is not None


def test_processing_pipeline_importable():
    """Test that ProcessingPipeline imports successfully."""
    from labanalysis.pipelines import ProcessingPipeline
    assert ProcessingPipeline is not None


def test_module_has_public_functions():
    """Test that pipelines module contains public functions."""
    public_attrs = [attr for attr in dir(pipelines) if not attr.startswith('_')]
    assert len(public_attrs) > 0
