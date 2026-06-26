# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [207] - 2026-06-26

### Documentation

- **BREAKING CLARITY**: Removed all incorrect axis letter (X/Y/Z) associations from code comments and documentation
- **bodies.py**: Updated all comments to use semantic axis names (lateral_axis, vertical_axis, anteroposterior_axis)
  - Updated 14 reference frame property docstrings
  - Updated 52 inline code comments
  - Updated 38 angle property docstrings
  - Removed all "Clinical Relevance" sections from angle docstrings
- **User guides**: Coordinate systems and joint angles documentation clarified
  - Added warnings distinguishing global vs local coordinate frames
  - Updated all examples to use semantic axis terminology
  - Removed misleading X/Y/Z axis associations
- **API reference**: ReferenceFrame and WholeBody documentation corrected
  - Emphasized column-based semantic axis mapping (Column 0 = lateral_axis, etc.)
  - Added "Semantic Axis Convention" explanations
  - Enhanced rotation matrix structure documentation
- **Tutorials and examples**: All code comments now reference semantic axes
  - Updated custom reference frames tutorial
  - Updated reference frames example code
- **Tests**: Comments clarified to distinguish global coordinate system conventions vs local reference frame semantics
  - Updated test_bodies_angles.py
  - Updated test_angle_signs.py
  - Updated test_scapular_angles.py
  - Updated test_referenceframe_migration.py

**Impact**: This is a documentation-only change. No code behavior has changed. Users should reference semantic axis names (lateral_axis, vertical_axis, anteroposterior_axis) rather than assuming fixed X/Y/Z letter mappings when working with reference frames. The rotation matrix column indices [0], [1], [2] correspond to lateral_axis, vertical_axis, and anteroposterior_axis respectively, as defined by the ReferenceFrame construction.

## [206] - 2026-06-25

### Fixed
- Corrected angle normalization and sign conventions for knee measurements
- Enhanced documentation and added tests to ensure angles remain within [-180°, +180°] range

### Added
- Comprehensive tests for ReferenceFrame class and its transformations
- Left/right lower and upper limb length properties to WholeBody class

### Changed
- Corrected varus/valgus angle calculations and sign conventions
- Updated README with version bump
- Improved code structure for better readability and maintainability
