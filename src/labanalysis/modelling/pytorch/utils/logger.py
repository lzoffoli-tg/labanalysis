"""Training logger for tracking model training progress."""

import sys
import time

import pandas as pd


class TrainingLogger:
    """
    Training logger for tracking model training progress.

    Handles logging of losses, metrics, learning rates, and early-stopping state
    during model training. Provides methods to update, query, and export training
    history.

    Attributes
    ----------
    _history : dict[str, list]
        Dictionary storing all logged values as lists.
    _best_loss : float
        Best validation loss observed so far.
    _epochs_without_improvement : int
        Number of consecutive epochs without improvement.
    _patience : int
        Total patience for early stopping.
    """

    def __init__(
        self,
        early_stopping_patience: int = 1000,
    ):
        """
        Initialize the TrainingLogger.

        Parameters
        ----------
        early_stopping_patience : int, default=1000
            Maximum number of epochs without improvement before early stopping.
        """
        self._history = {}
        self._best_loss = float("inf")
        self._epochs_without_improvement = 0
        self._patience = early_stopping_patience
        self._last_print_lines = 0  # Track lines printed for in-place updates
        self._last_minimal_length = 0  # Track length of last minimal output
        self._start_time = None  # Training start time

    def update(self, key: str, value: float):
        """
        Append a value to the specified metric key.

        Parameters
        ----------
        key : str
            Metric name (e.g., 'training_loss', 'validation_loss').
        value : float
            Value to append.
        """
        self._history.setdefault(key, []).append(value)

    def update_early_stopping_state(
        self, current_val_loss: float, threshold: float
    ):
        """
        Update early stopping state based on validation loss.

        Parameters
        ----------
        current_val_loss : float
            Current epoch's validation loss.
        threshold : float
            Minimum improvement threshold.

        Returns
        -------
        improved : bool
            True if validation loss improved beyond threshold.
        """
        if current_val_loss < self._best_loss - threshold:
            self._best_loss = current_val_loss
            self._epochs_without_improvement = 0
            return True
        else:
            self._epochs_without_improvement += 1
            return False

    def get_early_stopping_gap(self):
        """
        Get the remaining epochs before triggering early stopping.

        Returns
        -------
        gap : int
            Number of epochs remaining before early stopping is triggered.
        """
        return max(0, self._patience - self._epochs_without_improvement)

    def start_timer(self):
        """Start the training timer."""
        self._start_time = time.time()

    def get_elapsed_time(self):
        """
        Get formatted elapsed time since training started.

        Returns
        -------
        elapsed : str
            Formatted elapsed time (e.g., "1h 23m 45s" or "45s").
        """
        if self._start_time is None:
            return "0s"

        elapsed = time.time() - self._start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)

        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"

    def log_learning_rate(self, lr: float):
        """
        Log the current learning rate.

        Parameters
        ----------
        lr : float
            Current learning rate.
        """
        self.update("learning_rate", lr)

    def get_current_epoch(self):
        """
        Get the current epoch number.

        Returns
        -------
        epoch : int
            Current epoch number (0 if no epochs logged).
        """
        return len(self._history.get("epoch", []))

    def get_last_value(self, key: str):
        """
        Get the most recent value for a metric.

        Parameters
        ----------
        key : str
            Metric name.

        Returns
        -------
        value : float or None
            Most recent value, or None if key doesn't exist.
        """
        values = self._history.get(key)
        return values[-1] if values else None

    def to_dataframe(self):
        """
        Export training history to a pandas DataFrame.

        Returns
        -------
        df : pandas.DataFrame
            DataFrame with all logged metrics.
        """
        return pd.DataFrame(self._history)

    @property
    def history(self):
        """
        Access the full training history.

        Returns
        -------
        history : dict[str, list]
            Dictionary of all logged metrics.
        """
        return self._history

    @property
    def best_loss(self):
        """Best validation loss observed."""
        return self._best_loss

    @property
    def epochs_without_improvement(self):
        """Number of epochs without improvement."""
        return self._epochs_without_improvement

    def print_epoch_summary(
        self,
        epoch: int,
        val_loss: float,
        lr: float,
        verbose: str = "minimal",
    ):
        """
        Print a summary of the current epoch.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        val_loss : float
            Validation loss for this epoch.
        lr : float
            Current learning rate.
        verbose : {'full', 'minimal', 'off'}, default='minimal'
            Verbosity level:
            - 'full': Multi-line output with per-output losses and metrics (updates in place)
            - 'minimal': Single-line output with global losses and metrics (updates in place)
            - 'off': No output

        Notes
        -----
        - 'full' mode displays training and validation losses/metrics for each model output
        - 'minimal' mode displays:
            * Global loss: AVERAGE of all output losses (for interpretability and comparability)
            * Metrics: AVERAGE across all outputs (for multi-output models)
            * Note: optimization still uses sum/weighted sum for gradient computation
        - Both modes update in place using ANSI escape codes for clean terminal output
        - 'full' mode shows: epoch, learning rate, best loss, epochs without improvement, gap, and elapsed time
        - 'minimal' mode shows: epoch, training/validation losses, learning rate, gap, and elapsed time
        """
        if verbose == "off":
            return

        gap = self.get_early_stopping_gap()
        elapsed_time = self.get_elapsed_time()

        if verbose == "full":
            # Clear previous output (move cursor up and clear lines)
            if self._last_print_lines > 0:
                for _ in range(self._last_print_lines):
                    print("\033[F\033[K", end="")  # Move up and clear line

            # Extract all training and validation metrics
            lines = [
                f"{'='*80}",
                f"Epoch {epoch} | LR: {lr:.2e} | Best: {self._best_loss:.6f} | "
                f"No Improve: {self._epochs_without_improvement} | Gap: {gap} | Time: {elapsed_time}",
                f"{'-'*80}",
            ]

            # Get training loss
            train_loss = self.get_last_value("training_loss")
            if train_loss is not None:
                lines.append(f"Training Loss:   {train_loss:.6f}")

            # Get validation loss
            lines.append(f"Validation Loss: {val_loss:.6f}")

            # Find all output-specific metrics
            output_keys = set()
            metric_keys = set()
            for key in self._history.keys():
                if "_loss" in key and key not in ["training_loss", "validation_loss"]:
                    # Extract output name (e.g., "training_output1_loss" -> "output1")
                    parts = key.split("_")
                    if len(parts) >= 3:
                        output_name = "_".join(
                            parts[1:-1]
                        )  # everything between step_type and "loss"
                        output_keys.add(output_name)
                elif key.startswith("training_") or key.startswith("validation_"):
                    # Check if it's a metric
                    parts = key.split("_")
                    if len(parts) >= 3 and parts[-1] not in ["loss"]:
                        metric_name = parts[-1]
                        metric_keys.add(metric_name)

            # Display per-output metrics
            if output_keys:
                lines.append(f"{'-'*80}")
                for output_name in sorted(output_keys):
                    lines.append(f"\n{output_name.upper()}:")

                    # Loss
                    train_out_loss = self.get_last_value(f"training_{output_name}_loss")
                    val_out_loss = self.get_last_value(f"validation_{output_name}_loss")
                    if train_out_loss is not None and val_out_loss is not None:
                        lines.append(
                            f"  Loss:    train={train_out_loss:.6f}  val={val_out_loss:.6f}"
                        )

                    # Metrics
                    for metric_name in sorted(metric_keys):
                        train_metric = self.get_last_value(
                            f"training_{output_name}_{metric_name}"
                        )
                        val_metric = self.get_last_value(
                            f"validation_{output_name}_{metric_name}"
                        )
                        if train_metric is not None and val_metric is not None:
                            lines.append(
                                f"  {metric_name.upper()}:  train={train_metric:.6f}  val={val_metric:.6f}"
                            )

            lines.append(f"{'='*80}")

            # Join and print
            output_text = "\n".join(lines)
            print(output_text)

            # Track number of lines printed (count newlines + 1)
            self._last_print_lines = output_text.count("\n") + 1

        elif verbose == "minimal":
            # Get training loss (sum of all outputs, averaged over batches)
            train_loss = self.get_last_value("training_loss")
            train_str = f"train={train_loss:.4f}" if train_loss is not None else ""

            # Build metric strings (cached to avoid repeated dict lookups)
            metric_strs = []
            metric_keys = set()

            # Collect all per-output metrics to compute average
            output_metrics = {}  # {metric_name: {output_name: (train, val)}}

            for key in self._history.keys():
                if (
                    key.startswith("training_")
                    and not "_loss" in key
                    and not "learning_rate" in key
                ):
                    parts = key.split("_")
                    if len(parts) >= 3:  # training_{output}_{metric}
                        metric_name = parts[-1]
                        output_name = "_".join(parts[1:-1])

                        if metric_name not in output_metrics:
                            output_metrics[metric_name] = {}

                        train_val = self.get_last_value(key)
                        val_key = f"validation_{output_name}_{metric_name}"
                        val_val = self.get_last_value(val_key)

                        if train_val is not None and val_val is not None:
                            output_metrics[metric_name][output_name] = (train_val, val_val)

            # Compute average metrics across all outputs
            for metric_name in sorted(output_metrics.keys()):
                outputs_data = output_metrics[metric_name]
                if outputs_data:
                    # Average across outputs
                    avg_train = sum(t for t, v in outputs_data.values()) / len(outputs_data)
                    avg_val = sum(v for t, v in outputs_data.values()) / len(outputs_data)
                    metric_strs.append(
                        f"{metric_name}:t={avg_train:.4f}/v={avg_val:.4f}"
                    )

            metrics_part = " | ".join(metric_strs) if metric_strs else ""

            output = f"Epoch {epoch} | {train_str} | val={val_loss:.4f}"
            if metrics_part:
                output += f" | {metrics_part}"
            output += f" | lr={lr:.2e} | gap={gap} | time={elapsed_time}"

            # Use sys.stdout.write for faster I/O
            if self._last_minimal_length > 0:
                sys.stdout.write(" " * self._last_minimal_length + "\r")

            sys.stdout.write(output + "\r")
            sys.stdout.flush()  # Ensure immediate display
            self._last_minimal_length = len(output)
