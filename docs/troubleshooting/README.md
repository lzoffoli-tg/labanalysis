# Troubleshooting

Solutions to common issues and problems when using labanalysis.

## Quick Links

- **[Common Errors](common-errors.md)** - Frequently encountered errors and fixes
- **[Installation Issues](installation-issues.md)** - Problems installing labanalysis
- **[Data Loading Issues](data-loading-issues.md)** - File format and loading problems
- **[Performance Issues](performance-issues.md)** - Speed and optimization problems

## Most Common Issues

### 1. File Not Found

**Problem:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'data.tdf'
```

**Solutions:**

Use absolute path:
```python
import os
file_path = os.path.abspath("path/to/data.tdf")
record = laban.TimeseriesRecord.from_tdf(file_path)
```

Check file exists:
```python
from pathlib import Path

file_path = Path("data.tdf")
if file_path.exists():
    record = laban.TimeseriesRecord.from_tdf(file_path)
else:
    print(f"File not found: {file_path}")
```

### 2. Import Error

**Problem:**
```
ImportError: No module named 'labanalysis'
```

**Solutions:**

Verify installation:
```bash
pip list | grep labanalysis
```

Reinstall:
```bash
pip install --upgrade git+https://github.com/lzoffoli-tg/labanalysis.git
```

Check Python environment:
```bash
which python  # Linux/Mac
where python  # Windows
```

### 3. Missing Key

**Problem:**
```
KeyError: 'FP1'
```

**Solutions:**

Check available keys:
```python
record = laban.TimeseriesRecord.from_tdf("data.tdf")
print(f"Available keys: {list(record.keys())}")
```

Handle missing data:
```python
if 'FP1' in record:
    fp = record['FP1']
else:
    print("Force platform FP1 not found")
```

### 4. Version Conflicts

**Problem:**
```
ERROR: package-name requires dependency<version, but you have dependency>version
```

**Solutions:**

Create fresh virtual environment:
```bash
python -m venv labanalysis_env
source labanalysis_env/bin/activate  # Linux/Mac
labanalysis_env\Scripts\activate     # Windows
pip install git+https://github.com/lzoffoli-tg/labanalysis.git
```

### 5. Slow Performance

**Problem:** Processing takes too long

**Solutions:**

See [Performance Issues](performance-issues.md) for detailed optimization strategies.

Quick fixes:
- Use CPU optimization guide for PyTorch: [CPU_OPTIMIZATION_GUIDE.md](../advanced/CPU_OPTIMIZATION_GUIDE.md)
- Reduce data resolution
- Process in batches
- Use appropriate filter orders

## Getting Help

### Before Asking for Help

1. **Check this troubleshooting guide**
2. **Read the relevant user guide**: [User Guides](../user-guide/README.md)
3. **Check API reference**: [API Reference](../api-reference/README.md)
4. **Try the examples**: [Examples](../examples/README.md)

### Reporting Issues

When reporting an issue, include:

1. **Error message** - Complete traceback
2. **Code snippet** - Minimal reproducible example
3. **Environment** - Python version, OS, labanalysis version
4. **Data info** - File type, size, format (if relevant)

Example bug report:
```
**Issue:** Cannot load TDF file

**Error:**
FileNotFoundError: [Errno 2] No such file or directory: 'data.tdf'

**Code:**
import labanalysis as laban
record = laban.TimeseriesRecord.from_tdf("data.tdf")

**Environment:**
- Python 3.12.0
- Windows 11
- labanalysis version 202
- File: BTS TDF file, 50MB

**What I tried:**
- Verified file exists with os.path.exists()
- Tried absolute path
- File opens in BTS software
```

### Contact

For issues not covered in this guide:

**Email:** [lzoffoli@technogym.com](mailto:lzoffoli@technogym.com)

Include:
- Clear description of the problem
- Steps to reproduce
- Error messages
- System information

## Detailed Troubleshooting

- **[Common Errors](common-errors.md)** - Frequent errors and solutions
- **[Installation Issues](installation-issues.md)** - Installation and setup problems
- **[Data Loading Issues](data-loading-issues.md)** - File loading and format issues
- **[Performance Issues](performance-issues.md)** - Optimization and speed

---

**Still stuck?** Contact [lzoffoli@technogym.com](mailto:lzoffoli@technogym.com)
