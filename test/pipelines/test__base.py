"""Tests for pipelines._base module."""

import pytest


@pytest.mark.unit
class TestProcessingPipeline:
    """Test ProcessingPipeline class."""

    def test_module_imports(self):
        """Test ProcessingPipeline can be imported."""
        from labanalysis.pipelines._base import ProcessingPipeline
        assert ProcessingPipeline is not None

    def test_init_empty(self):
        """Test initialization with no callables."""
        from labanalysis.pipelines._base import ProcessingPipeline
        pipeline = ProcessingPipeline()
        assert len(pipeline.keys()) == 0

    def test_copy_creates_new_pipeline(self):
        """copy() creates a new ProcessingPipeline instance."""
        from labanalysis.pipelines._base import ProcessingPipeline

        def dummy_func(x):
            return x

        pipeline = ProcessingPipeline(Signal1D=dummy_func)
        pipeline_copy = pipeline.copy()

        assert isinstance(pipeline_copy, ProcessingPipeline)
        assert pipeline_copy is not pipeline
        assert pipeline_copy.keys() == pipeline.keys()

    def test_copy_preserves_callables(self):
        """copy() preserves all callable functions."""
        from labanalysis.pipelines._base import ProcessingPipeline

        def func1(x):
            return x

        def func2(x):
            return x * 2

        pipeline = ProcessingPipeline(Signal1D=func1, Signal3D=[func1, func2])
        pipeline_copy = pipeline.copy()

        assert pipeline_copy['Signal1D'] == pipeline['Signal1D']
        assert pipeline_copy['Signal3D'] == pipeline['Signal3D']

    def test_processing_pipeline_has_add_method(self):
        """Test ProcessingPipeline has add method."""
        from labanalysis.pipelines._base import ProcessingPipeline

        assert hasattr(ProcessingPipeline, 'add')
        assert callable(getattr(ProcessingPipeline, 'add'))

    def test_processing_pipeline_has_remove_method(self):
        """Test ProcessingPipeline has remove method."""
        from labanalysis.pipelines._base import ProcessingPipeline

        assert hasattr(ProcessingPipeline, 'remove')
        assert callable(getattr(ProcessingPipeline, 'remove'))

    def test_processing_pipeline_dict_interface(self):
        """Test ProcessingPipeline supports dict-like operations."""
        from labanalysis.pipelines._base import ProcessingPipeline

        def dummy_func(x):
            return x

        pipeline = ProcessingPipeline(Signal1D=dummy_func)

        # Should support keys(), values(), items()
        assert hasattr(pipeline, 'keys')
        assert hasattr(pipeline, 'values')
        assert hasattr(pipeline, 'items')
        assert 'Signal1D' in pipeline.keys()
