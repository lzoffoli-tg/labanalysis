# Running Test Assets

This directory is intended for running biomechanics test data (TDF files).

## Important Note

Large TDF files (>50MB) are **excluded from the repository** to comply with GitHub's file size limits.

## For Local Development

If you want to run the integration tests (`TestRunningExerciseIntegration`), place a running test TDF file here:

```
test/assets/running_test/running.tdf
```

The file should contain:
- Marker data (at minimum: lHeel, rHeel, lToe, rToe)
- Force platform data (optional, for kinetics tests)
- Sampling rate: recommended 100-500 Hz

## Running Tests Without TDF Files

The test suite includes comprehensive synthetic data tests that **do not require** real TDF files:
- `TestRunningExerciseKinematics` - Synthetic marker tests
- `TestRunningExerciseKinetics` - Synthetic force platform tests
- `TestRunningExerciseEdgeCases` - Edge case testing
- `TestRunningStepProperties` - Phase property tests

Integration tests that require TDF files will be automatically **skipped** if the file is not present.

## File Format

Expected TDF structure:
- Points3D markers with names: lHeel, rHeel, lToe, rToe, etc.
- ForcePlatform objects with origin (CoP), force, and torque data
- Consistent time indexing across all signals
