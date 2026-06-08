"""
Test runner for signalprocessing tests.

This script ensures the local version of labanalysis is used instead of the installed one.
"""

import sys
import os

# Add src directory to the path BEFORE any imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, 'src')
sys.path.insert(0, src_dir)

# Now run pytest
import pytest

if __name__ == '__main__':
    # Run the signalprocessing tests
    exit_code = pytest.main([
        os.path.join(current_dir, 'test_signalprocessing.py'),
        '-v',
        '--tb=short'
    ])
    sys.exit(exit_code)
