# labanalysis.messages

User interaction dialog functions.

**Source**: `src/labanalysis/messages.py`

## Overview

Simple dialog functions for user confirmation and input.

## Functions

### askyesno()

Ask yes/no question.

```python
def askyesno(title: str, message: str) -> bool
```

**Returns:** `True` if yes, `False` if no

**Example:**
```python
from labanalysis.messages import askyesno

if askyesno("Confirm", "Overwrite existing file?"):
    # Proceed
    pass
```

---

### askyesnocancel()

Ask yes/no/cancel question.

```python
def askyesnocancel(title: str, message: str) -> bool | None
```

**Returns:** `True` (yes), `False` (no), `None` (cancel)

**Example:**
```python
from labanalysis.messages import askyesnocancel

result = askyesnocancel("Save", "Save changes?")
if result is True:
    # Save
    pass
elif result is False:
    # Don't save
    pass
else:
    # Cancel
    pass
```

---

**Simple dialog functions for user interaction.**
