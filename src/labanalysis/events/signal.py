"""signal class module"""

from typing import Any, Callable


class Signal:
    """
    Lightweight signal-slot implementation inspired by PyQt's ``pyqtSignal``.

    A signal maintains a collection of connected callables (slots). When the
    signal is emitted, all connected slots are invoked in the order they were
    connected.

    Optionally, a signal signature can be declared through one or more types.
    When a signature is present, emitted arguments are validated against the
    expected types.

    Parameters
    ----------
    *types : type
        Expected types for emitted arguments.

    Examples
    --------
    Create and use a signal without type checking:

    >>> signal = Signal()
    >>> signal.connect(print)
    >>> signal.emit("Hello")
    Hello

    Create a typed signal:

    >>> signal = Signal(int, str)
    >>> signal.connect(lambda n, s: print(n, s))
    >>> signal.emit(42, "Done")
    42 Done
    """

    def __init__(self, *types: type) -> None:
        self._types: tuple[type, ...] = types
        self._slots: list[Callable[..., Any]] = []

    @property
    def types(self) -> tuple[type, ...]:
        """
        Return the declared signal signature.

        Returns
        -------
        tuple[type, ...]
            Expected argument types. An empty tuple indicates that no type
            validation is performed.
        """
        return self._types

    @property
    def slot_count(self) -> int:
        """
        Return the number of connected slots.

        Returns
        -------
        int
            Number of connected slots.
        """
        return len(self._slots)

    def connect(self, slot: Callable[..., Any]) -> None:
        """
        Connect a slot to the signal.

        A slot can be any callable object, including functions, methods,
        lambdas, and callable classes.

        Parameters
        ----------
        slot : Callable[..., Any]
            Callable to invoke when the signal is emitted.

        Raises
        ------
        TypeError
            If ``slot`` is not callable.
        """
        if not callable(slot):
            raise TypeError("slot must be callable")

        if slot not in self._slots:
            self._slots.append(slot)

    def disconnect(self, slot: Callable[..., Any]) -> None:
        """
        Disconnect a slot from the signal.

        Parameters
        ----------
        slot : Callable[..., Any]
            Slot to remove.

        Notes
        -----
        If the slot is not connected, the method silently returns.
        """
        try:
            self._slots.remove(slot)
        except ValueError:
            pass

    def emit(self, *args: Any) -> None:
        """
        Emit the signal.

        All connected slots are invoked sequentially with the provided
        arguments.

        Parameters
        ----------
        *args : Any
            Arguments forwarded to all connected slots.

        Raises
        ------
        TypeError
            If the emitted arguments do not match the declared signature.
        """
        self._validate_arguments(*args)

        # Iterate over a copy in case slots modify connections during emission.
        for slot in self._slots.copy():
            slot(*args)

    def clear(self) -> None:
        """
        Disconnect all connected slots.
        """
        self._slots.clear()

    def has_connections(self) -> bool:
        """
        Determine whether any slots are connected.

        Returns
        -------
        bool
            True if at least one slot is connected, otherwise False.
        """
        return bool(self._slots)

    def __call__(self, *args: Any) -> None:
        """
        Emit the signal using function-call syntax.

        Parameters
        ----------
        *args : Any
            Arguments passed to :meth:`emit`.

        Examples
        --------
        >>> signal = Signal(str)
        >>> signal.connect(print)
        >>> signal("Hello")
        Hello
        """
        self.emit(*args)

    def __len__(self) -> int:
        """
        Return the number of connected slots.

        Returns
        -------
        int
            Number of connected slots.
        """
        return len(self._slots)

    def __repr__(self) -> str:
        """
        Return a string representation of the signal.

        Returns
        -------
        str
            Human-readable representation.
        """
        signature = ", ".join(t.__name__ for t in self._types)
        return (
            f"{self.__class__.__name__}"
            f"(types=({signature}), slots={len(self._slots)})"
        )

    def _validate_arguments(self, *args: Any) -> None:
        """
        Validate emitted arguments against the signal signature.

        Parameters
        ----------
        *args : Any
            Arguments supplied to :meth:`emit`.

        Raises
        ------
        TypeError
            If the number or type of arguments does not match the declared
            signal signature.
        """
        if not self._types:
            return

        if len(args) != len(self._types):
            raise TypeError(
                f"Expected {len(self._types)} arguments, " f"got {len(args)}."
            )

        for index, (value, expected_type) in enumerate(
            zip(args, self._types),
            start=1,
        ):
            if not isinstance(value, expected_type):
                raise TypeError(
                    f"Argument {index} must be of type "
                    f"'{expected_type.__name__}', got "
                    f"'{type(value).__name__}'."
                )
