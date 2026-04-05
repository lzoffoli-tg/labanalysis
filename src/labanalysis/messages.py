"""
tk common messages module

this module provides an interface to the native message boxes available in Tk 4.2 and newer.

written by Fredrik Lundh, May 1997


options (all have default values):

    - default: which button to make default (one of the reply codes)
    - icon: which icon to display (see below)
    - message: the message to display
    - parent: which window to place the dialog on top of
    - title: dialog title
    - type: dialog type; that is, which buttons to display (see below)
"""

#! IMPORTS


from tkinter.commondialog import Dialog
import tkinter as tk

#! CONSTANTS

# icons
_ERROR = "error"
_INFO = "info"
_QUESTION = "question"
_WARNING = "warning"

# types
_OK = "ok"
_OKCANCEL = "okcancel"
_RETRYCANCEL = "retrycancel"
_YESNO = "yesno"
_YESNOCANCEL = "yesnocancel"

# replies
_RETRY = "retry"
_CANCEL = "cancel"
_YES = "yes"
_NO = "no"


#! CLASS


class _TkMessage(Dialog):
    """
    Tkinter message box dialog with topmost window behavior.

    Extended Dialog class that ensures message boxes appear on top of all
    windows and automatically creates a parent window if needed.

    Attributes
    ----------
    command : str
        Tkinter command name for message boxes.
    """

    command = "tk_messageBox"

    def show(self, **options):
        # Create a topmost parent window if it doesn't exist
        master = self.master
        if master is None:
            master = tk.Tk()
            master.withdraw()  # Hide the main window
            self.master = master

        # Bring the parent window to foreground and make it topmost
        try:
            master.attributes("-topmost", True)
            master.lift()
            master.focus_force()
        except tk.TclError:
            pass

        result = super().show(**options)

        # Clean up if we created the master window
        try:
            master.destroy()
        except tk.TclError:
            pass

        return result


#! METHODS


# Rename _icon and _type options to allow overriding them in options
def _show(title=None, message=None, _icon=None, _type=None, **options):
    """
    Internal function to display a Tkinter message box.

    Parameters
    ----------
    title : str, optional
        Dialog window title.
    message : str, optional
        Message text to display.
    _icon : str, optional
        Icon type ('error', 'info', 'question', 'warning').
    _type : str, optional
        Dialog type defining which buttons to show
        ('ok', 'okcancel', 'retrycancel', 'yesno', 'yesnocancel').
    **options : dict
        Additional Tkinter message box options.

    Returns
    -------
    str
        Button clicked by user ('ok', 'cancel', 'yes', 'no', 'retry').
    """
    if _icon and "icon" not in options:
        options["icon"] = _icon
    if _type and "type" not in options:
        options["type"] = _type
    if title:
        options["title"] = title
    if message:
        options["message"] = message
    res = _TkMessage(**options).show()
    # In some Tcl installations, yes/no is converted into a boolean.
    if isinstance(res, bool):
        if res:
            return _YES
        return _NO
    # In others we get a Tcl_Obj.
    return str(res)


def showinfo(title=None, message=None, **options):
    """
    Display an information message dialog.

    Shows a message box with an information icon and OK button.

    Parameters
    ----------
    title : str, optional
        Dialog window title.
    message : str, optional
        Information message to display.
    **options : dict
        Additional Tkinter message box options (parent, default, etc.).

    Returns
    -------
    str
        Always returns 'ok'.
    """
    return _show(title, message, _INFO, _OK, **options)


def showwarning(title=None, message=None, **options):
    """
    Display a warning message dialog.

    Shows a message box with a warning icon and OK button.

    Parameters
    ----------
    title : str, optional
        Dialog window title.
    message : str, optional
        Warning message to display.
    **options : dict
        Additional Tkinter message box options (parent, default, etc.).

    Returns
    -------
    str
        Always returns 'ok'.
    """
    return _show(title, message, _WARNING, _OK, **options)


def showerror(title=None, message=None, **options):
    """
    Display an error message dialog.

    Shows a message box with an error icon and OK button.

    Parameters
    ----------
    title : str, optional
        Dialog window title.
    message : str, optional
        Error message to display.
    **options : dict
        Additional Tkinter message box options (parent, default, etc.).

    Returns
    -------
    str
        Always returns 'ok'.
    """
    return _show(title, message, _ERROR, _OK, **options)


def askquestion(title=None, message=None, **options):
    """
    Ask a yes/no question dialog.

    Shows a message box with a question icon and Yes/No buttons.

    Parameters
    ----------
    title : str, optional
        Dialog window title.
    message : str, optional
        Question to ask the user.
    **options : dict
        Additional Tkinter message box options (parent, default, etc.).

    Returns
    -------
    str
        'yes' if Yes clicked, 'no' if No clicked.
    """
    return _show(title, message, _QUESTION, _YESNO, **options)


def askokcancel(title=None, message=None, **options):
    """
    Ask for confirmation to proceed with an operation.

    Shows a message box with a question icon and OK/Cancel buttons.

    Parameters
    ----------
    title : str, optional
        Dialog window title.
    message : str, optional
        Question or confirmation prompt.
    **options : dict
        Additional Tkinter message box options (parent, default, etc.).

    Returns
    -------
    bool
        True if OK clicked, False if Cancel clicked.
    """
    s = _show(title, message, _QUESTION, _OKCANCEL, **options)
    return s == _OK


def askyesno(title=None, message=None, **options):
    """
    Ask a yes/no question and return boolean result.

    Shows a message box with a question icon and Yes/No buttons.

    Parameters
    ----------
    title : str, optional
        Dialog window title.
    message : str, optional
        Question to ask the user.
    **options : dict
        Additional Tkinter message box options (parent, default, etc.).

    Returns
    -------
    bool
        True if Yes clicked, False if No clicked.
    """
    s = _show(title, message, _QUESTION, _YESNO, **options)
    return s == _YES


def askyesnocancel(title=None, message=None, **options):
    """
    Ask a yes/no/cancel question and return boolean or None.

    Shows a message box with a question icon and Yes/No/Cancel buttons.

    Parameters
    ----------
    title : str, optional
        Dialog window title.
    message : str, optional
        Question to ask the user.
    **options : dict
        Additional Tkinter message box options (parent, default, etc.).

    Returns
    -------
    bool or None
        True if Yes clicked, False if No clicked, None if Cancel clicked.
    """
    s = _show(title, message, _QUESTION, _YESNOCANCEL, **options)
    # s might be a Tcl index object, so convert it to a string
    s = str(s)
    if s == _CANCEL:
        return None
    return s == _YES


def askretrycancel(title=None, message=None, **options):
    """
    Ask whether to retry a failed operation.

    Shows a message box with a warning icon and Retry/Cancel buttons.

    Parameters
    ----------
    title : str, optional
        Dialog window title.
    message : str, optional
        Description of the operation to retry.
    **options : dict
        Additional Tkinter message box options (parent, default, etc.).

    Returns
    -------
    bool
        True if Retry clicked, False if Cancel clicked.
    """
    s = _show(title, message, _WARNING, _RETRYCANCEL, **options)
    return s == _RETRY


__all__ = [
    "showinfo",
    "showwarning",
    "showerror",
    "askquestion",
    "askokcancel",
    "askyesno",
    "askyesnocancel",
    "askretrycancel",
]
