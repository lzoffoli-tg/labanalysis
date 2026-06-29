# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [214] - 2026-06-29

### Fixed
- **EMGSignal copy behavior**: Fixed critical bug where `muscle_name` and `side` attributes were lost during slice operations
  - Modified `Timeseries._view()` to preserve `_name` and `_side` attributes when creating views/slices
  - This fixes the issue in `RunningExercise._get_cycle()` where EMG signals were copied and sliced
  - Affects all code using the pattern: `emg.copy()[start:stop]` or `emg[start:stop].copy()`
- **EMGSignal side validation**: Fixed typo in `set_side()` method ("bilataral" → "bilateral")
  - Now correctly accepts `side="bilateral"` parameter

### Added
- Comprehensive test suite for EMGSignal copy behavior (`test/timeseries/test_emgsignal_copy.py`)
- Integration tests for EMGSignal in exercise context (`test/exercises/gait/test_running_exercise_emg_copy.py`)
- Bug fix documentation (`test/BUGFIX_SUMMARY_EMGSignal_copy.md`)

### Technical Details
- **Files Modified**:
  - `src/labanalysis/timeseries/_base.py`: Lines 494-497 (added attribute preservation)
  - `src/labanalysis/timeseries/emgsignal.py`: Line 84 (fixed typo)
- **Test Coverage**: 14 new tests, all existing 80 timeseries tests still passing

## [213] - Previous Version

Initial tracked version.
