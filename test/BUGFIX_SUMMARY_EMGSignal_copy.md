# Bug Fix Summary: EMGSignal Copy Behavior

## Issue Description

When copying a `RunningExercise` object that contains `EMGSignal` instances, the `muscle_name` attribute was lost, causing `AttributeError` when accessed.

## Root Causes

### 1. **Main Issue: `_view` method doesn't preserve EMG-specific attributes**
   - **Location**: `src/labanalysis/timeseries/_base.py:484-493`
   - **Problem**: When creating a view/slice of a Timeseries object, only certain attributes (`_unit`, `_vertical_axis`, `_anteroposterior_axis`) were preserved
   - **Impact**: EMGSignal-specific attributes (`_name`, `_side`) were lost during slicing operations

### 2. **Typo in `set_side` method**
   - **Location**: `src/labanalysis/timeseries/emgsignal.py:84`
   - **Problem**: Typo "bilataral" instead of "bilateral" in validation
   - **Impact**: Unable to create EMGSignal with "bilateral" side

## How the Bug Manifested

In `RunningExercise._get_cycle()` (line 211):
```python
args.update(**{i: v.copy()[start:stop] for i, v in self.items()})
```

This pattern:
1. Calls `copy()` on each signal (including EMGSignal)
2. Then slices the result with `[start:stop]`
3. The slice operation calls `_view()` which creates a new object
4. The new object doesn't have `_name` or `_side` attributes
5. Accessing `muscle_name` property fails with `AttributeError`

## Files Modified

### 1. `src/labanalysis/timeseries/_base.py`
**Change**: Added preservation of `_name` and `_side` attributes in `_view` method

```python
# Lines 490-493 (added):
if hasattr(self, '_name'):
    view_obj._name = self._name
if hasattr(self, '_side'):
    view_obj._side = self._side
```

### 2. `src/labanalysis/timeseries/emgsignal.py`
**Change**: Fixed typo in `set_side` method

```python
# Line 84 (before):
[side == i for i in ["left", "right", "bilataral"]]

# Line 84 (after):
[side == i for i in ["left", "right", "bilateral"]]
```

## Tests Added

### 1. `test/timeseries/test_emgsignal_copy.py` (New File)
Comprehensive test suite covering:
- Basic copy operations
- Attribute preservation (muscle_name, side)
- Copy independence
- Sliced signal copying
- Different unit types
- Integration with dictionary copying (as used in RunningExercise)

**Key Tests**:
- `test_basic_copy`: Verifies basic copy functionality
- `test_copy_preserves_muscle_name`: Ensures muscle_name survives copy
- `test_sliced_copy`: Tests the exact failure scenario (slice then copy)
- `test_emg_time_slicing_and_copy`: Simulates RunningExercise._get_cycle pattern

### 2. `test/exercises/gait/test_running_exercise_emg_copy.py` (New File)
Exercise-specific tests covering:
- EMG signal slicing and copying pattern used in `_get_cycle`
- TimeseriesRecord dictionary iteration with EMG signals
- Multiple sequential copy/slice operations
- Both operation orders: `copy()[start:stop]` vs `[start:stop].copy()`

## Test Results

All tests pass:
- `test/timeseries/test_emgsignal_copy.py`: **10/10 passed**
- `test/exercises/gait/test_running_exercise_emg_copy.py`: **4/4 passed**
- All existing timeseries tests: **80/80 passed** (no regressions)

## Verification

The fix ensures that when `RunningExercise` creates gait cycles via `_get_cycle()`:
1. EMG signals are correctly sliced to the cycle time range
2. Muscle name and side information are preserved
3. All other EMG properties remain intact
4. The pattern `v.copy()[start:stop]` works correctly for EMGSignal objects

## Behavior in Different Conditions

### Condition 1: Direct Copy
```python
emg = EMGSignal(data, index, "biceps", "left")
emg_copy = emg.copy()
# ✓ muscle_name preserved
# ✓ side preserved
```

### Condition 2: Slice Then Copy
```python
emg_sliced = emg[2.0:4.0]
emg_copy = emg_sliced.copy()
# ✓ muscle_name preserved (after fix)
# ✓ side preserved (after fix)
```

### Condition 3: Copy Then Slice (RunningExercise pattern)
```python
emg_result = emg.copy()[2.0:4.0]
# ✓ muscle_name preserved (after fix)
# ✓ side preserved (after fix)
```

### Condition 4: Multiple Operations
```python
emg_multi = emg.copy()[2.0:8.0].copy()[4.0:6.0]
# ✓ muscle_name preserved (after fix)
# ✓ side preserved (after fix)
```

## Refactoring to Extensible System (v215)

The initial fix (v214) used hardcoded `hasattr` checks in `_view()`. This was refactored (v215) to use a hook-based system:

### Hook Method Pattern
- Added `_copy_view_attributes(view_obj)` method to `Timeseries` base class
- Subclasses override this method to preserve their custom attributes
- More maintainable and extensible than hardcoded checks

### Example Override
```python
class EMGSignal(Signal1D):
    def _copy_view_attributes(self, view_obj):
        super()._copy_view_attributes(view_obj)
        # Custom attributes are now handled by parent's default implementation
```

### Benefits
1. **Extensible**: New subclasses can easily add custom attributes
2. **Maintainable**: No need to modify base class for new attributes
3. **Documented**: Clear pattern for future developers
4. **Backward Compatible**: All existing code continues to work

See `test/timeseries/test_view_attribute_extensibility.py` for examples of custom subclasses.
