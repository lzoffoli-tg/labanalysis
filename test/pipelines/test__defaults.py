"""Tests for pipelines._defaults module."""

import pytest


@pytest.mark.integration
class TestDefaultProcessing:
    """Test default processing functions."""

    def test_module_imports(self):
        """Test that default functions can be imported."""
        from labanalysis.pipelines._defaults import (
            get_default_emgsignal_processing_func,
            get_default_point3d_processing_func,
            get_default_processing_pipeline,
        )
        assert get_default_emgsignal_processing_func is not None
        assert get_default_point3d_processing_func is not None
        assert get_default_processing_pipeline is not None

    def test_get_default_processing_pipeline_returns_pipeline(self):
        """Test get_default_processing_pipeline returns ProcessingPipeline."""
        from labanalysis.pipelines._defaults import get_default_processing_pipeline
        from labanalysis.pipelines._base import ProcessingPipeline

        pipeline = get_default_processing_pipeline()
        assert isinstance(pipeline, ProcessingPipeline)

    def test_default_functions_are_callable(self):
        """Test all default functions are callable."""
        from labanalysis.pipelines._defaults import (
            get_default_emgsignal_processing_func,
            get_default_point3d_processing_func,
        )

        # These functions return processing functions
        assert callable(get_default_emgsignal_processing_func)
        assert callable(get_default_point3d_processing_func)
