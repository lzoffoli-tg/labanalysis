# Changelog

All notable changes to this project will be documented in this file.

## [216] - 2026-06-29

### Fixed
- **Critical**: Fixed `EMGSignal.copy()` failing after `strip()` or slicing operations
  - `EMGSignal.copy()` now uses internal attributes (`_name`, `_side`) instead of properties
  - Prevents `AttributeError` when copying EMGSignal objects that lost attributes during slicing
- **Critical**: Fixed `Record.strip(independent=False, inplace=True)` losing subclass attributes
  - Now modifies Timeseries objects in-place instead of replacing them with views
  - Preserves all subclass-specific attributes (`_name`, `_side`, `_vertical_axis`, etc.)
  - Fixes issue where `copy()` would fail after `strip()` on TimeseriesRecord containing EMGSignal or Point3D

### Performance
- **Major**: Optimized `Timeseries.strip()` performance - **273x faster**
  - Before: ~1684 ms for 10,000 rows × 50 columns
  - After: ~6 ms for same dataset
  - Avoids creating DataFrame multiple times
  - Uses direct NumPy slicing instead of `.loc[]` indexing
  
- **Major**: Optimized `Record.strip(independent=False)` performance - **107x faster**
  - Before: ~4986 ms for 5 timeseries elements
  - After: ~46 ms for same dataset
  - Avoids creating giant concatenated DataFrame
  - Iterates over elements to find bounds instead of combining all data

### Tests
- Added comprehensive test suite for `strip()` optimizations
  - `test/timeseries/test_strip_optimization.py` - 10 tests
  - `test/records/records/test_strip_optimization.py` - 6 tests
- All tests verify both correctness and attribute preservation

### Technical Details
- `Timeseries.strip()`: Uses `np.isnan()` directly on `_data` when possible, creates DataFrame only when necessary
- `Record.strip()`: Modifies objects in-place using `object.__setattr__()` to bypass custom `__setattr__` in subclasses
- Handles nested Record objects (like ForcePlatform) recursively

## [215] - Previous version
- Extensible attribute preservation system for Timeseries
- Fix EMGSignal attribute loss during copy/slice operations
