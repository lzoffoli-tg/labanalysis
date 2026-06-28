"""Tests for pytorch.utils.logger module."""

import pytest


@pytest.mark.pytorch
class TestTrainingLogger:
    """Test TrainingLogger class."""

    def test_module_imports(self):
        """Test module can be imported."""
        from labanalysis.modelling.pytorch.utils.logger import TrainingLogger
        assert TrainingLogger is not None

    def test_training_logger_has_update_method(self):
        """Test TrainingLogger has update method."""
        from labanalysis.modelling.pytorch.utils.logger import TrainingLogger

        assert hasattr(TrainingLogger, 'update')
        assert callable(getattr(TrainingLogger, 'update'))

    def test_training_logger_has_to_dataframe_method(self):
        """Test TrainingLogger has to_dataframe method."""
        from labanalysis.modelling.pytorch.utils.logger import TrainingLogger

        assert hasattr(TrainingLogger, 'to_dataframe')
        assert callable(getattr(TrainingLogger, 'to_dataframe'))

    def test_training_logger_docstring_exists(self):
        """Test TrainingLogger has comprehensive docstring."""
        from labanalysis.modelling.pytorch.utils.logger import TrainingLogger

        assert TrainingLogger.__doc__ is not None
        assert len(TrainingLogger.__doc__) > 100
        assert 'logger' in TrainingLogger.__doc__.lower() or 'training' in TrainingLogger.__doc__.lower()
