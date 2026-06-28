"""Trainer class for PyTorch models with advanced features."""

from typing import Callable, Literal

import torch
from torch.utils.data import DataLoader

from ....utils import split_data
from .datasets import CustomDataset, UncertaintyWeighting
from .losses import PinballLoss, QuantilicRangeLoss, ComboLoss
from .metrics import MAEMetric
from .logger import TrainingLogger


class TorchTrainer:
    """
    Trainer class for PyTorch models with support for multi-output training,
    early stopping, learning rate decay, metric tracking, and optional
    Uncertainty Weighting (Kendall & Gal, CVPR 2018).

    This trainer handles:
    - multi-output predictions (dict-based)
    - internal optimizer initialization
    - optional Uncertainty Weighting for automatic task balancing
    - early stopping with patience
    - learning rate decay after stagnation
    - detailed per-output and global logging of losses/metrics
    - validation split management and best weight restoration

    Parameters
    ----------
    loss : callable
        Loss function taking (y_pred, y_true) → scalar tensor.

    metrics : callable or list or dict
        Metric functions evaluated on predictions each epoch.

    optimizer_class : type(torch.optim.Optimizer), default=torch.optim.AdamW
        Class of the optimizer to create internally. AdamW is recommended for
        most cases as it generally provides better generalization than Adam.

    optimizer_kwargs : dict, optional
        Additional keyword arguments for optimizer construction.
        Example: {"lr": 1e-3, "weight_decay": 1e-5}
        The 'lr' parameter can be a single float or a list of floats.
        If a list is provided, learning rates are applied sequentially:
        when early_stopping_patience is exceeded, the next learning rate
        is applied and weights are restored to the best state found so far.

    epochs : int, default=100000
        Maximum number of epochs.

    batch_size : int, default=256
        Batch size for training. Default of 256 is optimal for CPU vectorization.
        Set to None to use full batch. Larger values (128-512) generally perform
        better on CPU due to better vectorization.

    early_stopping_threshold : float, default=1e-5
        Minimum delta in validation loss to reset patience.

    early_stopping_patience : int, default=2000
        Max number of epochs without improvement.

    validation_split : float, default=0.2
        Fraction of data used for validation.

    restore_best_weights : bool, default=True
        Whether to restore best model after early stopping.

    verbose : {'full', 'minimal', 'off'}, default='minimal'
        Verbosity level during training.

    debug : bool, default=False
        Enables gradient sanity checks.

    use_uncertainty_weighting : bool, default=False
        Whether to apply task uncertainty weighting.

    num_workers : int or None, default=None
        Number of worker processes for data loading. If None (default), automatically
        selects the optimal value based on OS, CPU count, and dataset size.
        Auto-tuning logic:
        - Windows: uses 0 (multiprocessing can be unstable)
        - Linux/Mac with small dataset (<1000 samples): uses 0 (overhead not worth it)
        - Linux/Mac with large dataset: uses min(cpu_count // 2, 4)
        Set explicitly to override auto-tuning (e.g., 0 to disable, 2-4 for manual control).

    use_torch_compile : bool, default=True
        Whether to use torch.compile() for model optimization (PyTorch 2.0+).
        Provides ~50-100% speedup. Automatically disabled if not available.
        Requires PyTorch >= 2.0 and C++ compiler (MSVC on Windows, gcc/clang on Linux/Mac).
        On Windows without MSVC, automatically falls back to non-compiled mode.

    gradient_clip_val : float, default=1.0
        Maximum gradient norm for gradient clipping. Default of 1.0 prevents
        exploding gradients in most cases. Set to None to disable clipping.
        Increase to 5.0-10.0 for problems that need larger gradients.

    use_fused_optimizer : bool, default=True
        Whether to use fused optimizer kernels for Adam/AdamW (if available).
        Can provide ~10-15% speedup for optimizer step. Automatically disabled
        if not supported by the optimizer or PyTorch version.

    ema_decay : float or None, default=None
        Exponential Moving Average decay rate for model weights (e.g., 0.999).
        If set, maintains EMA weights and uses them for validation and best weights.
        Improves model stability and generalization. Recommended: 0.999 or 0.9999.

    gradient_accumulation_steps : int, default=1
        Number of batches to accumulate gradients before optimizer step.
        Simulates larger batch sizes without increasing memory usage.
        Effective batch size = batch_size * gradient_accumulation_steps.

    Attributes
    ----------
    logger : dict[str, list]
        Stores all training logs (losses, metrics, epochs).

    _uw_module : UncertaintyWeighting or None
        Internal module providing learnable loss weights.

    Notes
    -----
    **Basic Usage:**
    - Models must return either a dict or a tensor.
    - Extra losses must be functions with no parameters.
    - Multi-output handling is native and automatic.
    - Learning rate scheduling: If optimizer_kwargs["lr"] is a list, the trainer will
      automatically switch to the next learning rate when early_stopping_patience is
      exceeded, restoring the best weights found so far. Training stops when all
      learning rates have been tried or max epochs is reached.

    **Performance Tips:**
    - Default settings are optimized for CPU training with ~2.5-3.5x speedup over baseline
    - num_workers is auto-tuned by default (considers OS, CPU count, dataset size)
    - For memory-limited systems: reduce batch_size to 128 or use gradient_accumulation_steps
    - For better stability: enable ema_decay=0.999 (costs 2x memory for weights)
    - For maximum speed: ensure PyTorch >= 2.0 for torch.compile support

    **Common Configurations:**
    - Fast & stable: use defaults (batch_size=256, gradient_clip_val=1.0, auto-tuned num_workers)
    - Memory limited: batch_size=64, gradient_accumulation_steps=4
    - Maximum stability: ema_decay=0.999, gradient_clip_val=1.0
    - Maximum speed: use defaults (all optimizations enabled, num_workers auto-tuned)
    """

    def __init__(
        self,
        loss=ComboLoss(PinballLoss(0.5), QuantilicRangeLoss(0.99)),
        metrics=[MAEMetric()],
        optimizer_class=torch.optim.AdamW,  # AdamW generalmente migliore di Adam
        optimizer_kwargs={"lr": [1e-3, 1e-4, 1e-5]},
        epochs: int = 100000,
        batch_size: int | None = 256,  # Valore ottimale per CPU vectorization
        early_stopping_threshold: float = 1e-5,
        early_stopping_patience: int = 200,
        validation_split: float = 0.2,
        restore_best_weights: bool = True,
        verbose: Literal["full", "minimal", "off"] = "minimal",
        debug: bool = False,
        use_uncertainty_weighting: bool = True,
        num_workers: int | None = None,  # None = auto-tune based on OS/dataset/CPU
        use_torch_compile: bool = True,  # Requires PyTorch 2.0+
        gradient_clip_val: float | None = 1.0,  # Prevents exploding gradients
        use_fused_optimizer: bool = True,  # ~10-15% speedup for Adam/AdamW
        ema_decay: float | None = None,  # Set to 0.999 for better stability (doubles memory)
        gradient_accumulation_steps: int = 1,  # >1 simulates larger batch sizes
    ):

        # ------------ BASIC VALIDATION ------------
        if batch_size is not None and not isinstance(batch_size, int):
            raise ValueError("'batch_size' must be int or None.")

        self._batch_size = batch_size
        self._loss = loss
        self._optimizer_class = optimizer_class
        self._optimizer_kwargs = optimizer_kwargs or {}
        self._epochs = epochs

        self._early_stopping_threshold = early_stopping_threshold
        self._early_stopping_patience = early_stopping_patience

        self._validation_split = validation_split
        self._restore_best_weights = restore_best_weights
        self._verbose = verbose
        self._debug = debug
        self._num_workers = num_workers
        self._use_torch_compile = use_torch_compile
        self._gradient_clip_val = gradient_clip_val
        self._use_fused_optimizer = use_fused_optimizer
        self._ema_decay = ema_decay
        self._gradient_accumulation_steps = gradient_accumulation_steps
        self._ema_state = None  # Will store EMA weights if enabled
        self._compile_failed = False  # Track if torch.compile failed during execution

        # ------------ LEARNING RATE SCHEDULING ------------
        # Handle both single lr and list of lrs
        lr_param = self._optimizer_kwargs.get("lr", 0.001)
        if isinstance(lr_param, list):
            self._learning_rates = lr_param.copy()
            self._current_lr_index = 0
        else:
            self._learning_rates = [lr_param]
            self._current_lr_index = 0

        # ------------ METRICS ------------
        if isinstance(metrics, Callable):
            self._metrics = {metrics.__class__.__name__.lower(): metrics}
        elif isinstance(metrics, list):
            self._metrics = {m.__class__.__name__.lower(): m for m in metrics}
        elif isinstance(metrics, dict):
            self._metrics = metrics
        else:
            raise ValueError("metrics must be callable, list, or dict.")

        # ------------ UNCERTAINTY WEIGHTING ------------
        self._use_uncertainty_weighting = use_uncertainty_weighting
        self._uw_module = None

        # ------------ LOGGER ------------
        self._logger = TrainingLogger(
            early_stopping_patience=early_stopping_patience,
        )

    # ==================================================================
    # TORCH.COMPILE AVAILABILITY CHECK
    # ==================================================================

    @staticmethod
    def _is_torch_compile_available() -> bool:
        """
        Check if torch.compile() is available and usable.

        Returns
        -------
        available : bool
            True if torch.compile() can be used, False otherwise.

        Notes
        -----
        torch.compile() requires:
        - PyTorch >= 2.0
        - C++ compiler in PATH on Windows (MSVC cl.exe)
        - C++ compiler on Linux/Mac (gcc/clang, usually available)

        On Windows, the MSVC compiler must be accessible via system PATH.
        If Visual Studio is installed but compiler is not in PATH, torch.compile
        will not work. Use scripts/add_compiler_to_path.ps1 to add it.
        """
        import platform
        import shutil

        # Check if torch.compile exists (PyTorch 2.0+)
        if not hasattr(torch, "compile"):
            return False

        # On Windows, check if MSVC compiler (cl.exe) is available in PATH
        # PyTorch's torch.compile requires cl.exe to be accessible via PATH
        if platform.system() == "Windows":
            return shutil.which("cl") is not None

        # On Linux/Mac, torch.compile usually works (gcc/clang often present)
        # If compiler is missing, torch.compile will fall back gracefully
        return True

    # ==================================================================
    # NUM_WORKERS AUTO-TUNING
    # ==================================================================

    @staticmethod
    def _compute_optimal_num_workers(dataset_size: int) -> int:
        """
        Compute optimal number of DataLoader workers based on OS, CPU, and dataset size.

        Parameters
        ----------
        dataset_size : int
            Number of samples in the dataset.

        Returns
        -------
        num_workers : int
            Optimal number of worker processes for data loading.

        Notes
        -----
        Auto-tuning logic:
        - Windows: always returns 0 (multiprocessing can be unstable)
        - Small datasets (<1000 samples): returns 0 (overhead not worth it)
        - Large datasets: returns min(cpu_count // 2, 4)
          Using half of available CPUs leaves resources for main process and other tasks.
          Capping at 4 prevents excessive context switching and memory overhead.
        """
        import os
        import platform

        # Windows: multiprocessing is unstable, use 0
        if platform.system() == "Windows":
            return 0

        # Small datasets: overhead not worth it
        if dataset_size < 1000:
            return 0

        # Large datasets: use half of CPUs, max 4
        cpu_count = os.cpu_count() or 1
        return min(cpu_count // 2, 4)

    # ==================================================================
    # LOGGING HELPERS
    # ==================================================================

    @property
    def logger(self):
        """Return training metrics history."""
        return self._logger.history

    def _update_logger(self, key: str, value: float):
        """Append a value to the given metric key."""
        self._logger.update(key, value)

    # ==================================================================
    # EMA HELPERS
    # ==================================================================

    def _update_ema(self, module):
        """
        Update Exponential Moving Average of model weights.

        Parameters
        ----------
        module : torch.nn.Module
            Model whose weights to track with EMA (may be compiled).
        """
        if self._ema_state is None:
            return

        # Always use original module for parameter names (compiled module may have different names)
        with torch.no_grad():
            for name, param in self._original_module.named_parameters():
                if param.requires_grad:
                    self._ema_state[name].mul_(self._ema_decay).add_(
                        param.data, alpha=1 - self._ema_decay
                    )

    def _apply_ema(self, module):
        """Apply EMA weights to module (for validation/best weights)."""
        if self._ema_state is None:
            return None

        # Always use original module for EMA (consistent with parameter names)
        # Since torch.compile shares parameters with original, modifying original affects compiled too
        original_state = {k: v.clone() for k, v in self._original_module.state_dict().items()}
        self._original_module.load_state_dict(self._ema_state)
        return original_state

    def _restore_original_weights(self, module, original_state):
        """Restore original weights after EMA application."""
        if original_state is not None:
            self._original_module.load_state_dict(original_state)

    # ==================================================================
    # BATCH PROCESSING
    # ==================================================================

    def _process_batch(self, module, x_batch, y_batch, extra_losses=[]):
        """
        Compute predictions, mask invalid samples, compute per-output losses.

        Parameters
        ----------
        module : torch.nn.Module
            Model to evaluate.

        x_batch, y_batch : dict[str, Tensor] or Tensor
            Mini-batch inputs/targets.

        extra_losses : list[callable]
            Additional regularization terms.

        Returns
        -------
        batch_trues, batch_preds : dict[str, Tensor] or Tensor
            Valid samples only.

        batch_losses : dict[str, Tensor] or Tensor
            Per-output loss values.

        batch_samples : dict[str, int] or int
            Counts of valid samples.
        """

        preds = module(x_batch)
        was_dict = True

        if not isinstance(preds, dict):
            preds = {"output": preds}
            was_dict = False
        if not isinstance(y_batch, dict):
            y_batch = {"output": y_batch}
            was_dict = False

        batch_losses = {}
        batch_trues = {}
        batch_preds = {}
        batch_samples = {}

        for key in y_batch:
            mask = torch.isfinite(y_batch[key]) & torch.isfinite(preds[key])
            t = y_batch[key][mask]
            p = preds[key][mask]

            batch_trues[key] = t
            batch_preds[key] = p
            batch_samples[key] = mask.sum().item()  # Convert to int immediately
            batch_losses[key] = self._loss(p, t)

        for extra_loss in extra_losses:
            batch_losses[extra_loss.__name__] = extra_loss()

        if not was_dict:
            return (
                list(batch_trues.values())[0],
                list(batch_preds.values())[0],
                list(batch_losses.values())[0],
                list(batch_samples.values())[0],
            )

        return batch_trues, batch_preds, batch_losses, batch_samples

    # ==================================================================
    # STEP (TRAIN OR VALIDATION)
    # ==================================================================

    def _step(self, module, loader, step_type, extra_losses=[]):
        """
        Execute one full epoch pass (train or validation).

        Parameters
        ----------
        module : torch.nn.Module
        loader : DataLoader
        step_type : {"training", "validation"}
        extra_losses : list

        Notes
        -----
        - Applies UW weighting if active.
        - Logs per-output and global losses.
        - Computes all metrics incrementally for better memory efficiency.
        """

        is_dict = isinstance(loader.dataset.y, dict)

        if is_dict:
            keys = list(loader.dataset.y.keys())
            losses = {k: 0.0 for k in keys}
            samples = {k: 0 for k in keys}
            # Incremental metric computation
            metric_accumulators = {
                k: {mname: 0.0 for mname in self._metrics.keys()} for k in keys
            }
        else:
            losses = {"output": 0.0}
            samples = {"output": 0}
            metric_accumulators = {"output": {mname: 0.0 for mname in self._metrics.keys()}}

        total_loss = 0.0
        batches = 0
        accumulation_counter = 0

        for xb, yb in loader:
            bt, bp, bl, bs = self._process_batch(module, xb, yb, extra_losses)

            if isinstance(bl, dict):
                if self._use_uncertainty_weighting and self._uw_module:
                    batch_loss = self._uw_module(bl)
                else:
                    batch_loss = sum(bl.values())
            else:
                batch_loss = bl

            if step_type == "training":
                # Gradient accumulation: scale loss by accumulation steps
                scaled_loss = batch_loss / self._gradient_accumulation_steps

                # Zero gradients only at start of accumulation cycle
                if accumulation_counter == 0:
                    self._optimizer.zero_grad(set_to_none=True)

                scaled_loss.backward()
                accumulation_counter += 1

                # Update weights only after accumulating gradients
                if accumulation_counter >= self._gradient_accumulation_steps:
                    # Gradient clipping if enabled
                    if self._gradient_clip_val is not None:
                        torch.nn.utils.clip_grad_norm_(
                            module.parameters(), self._gradient_clip_val
                        )

                    self._optimizer.step()

                    # Update EMA if enabled
                    if self._ema_decay is not None:
                        self._update_ema(module)

                    accumulation_counter = 0

            with torch.no_grad():
                # Accumulate losses and compute metrics incrementally
                if isinstance(bl, dict):
                    for k, lv in bl.items():
                        v = lv.item()
                        n = bs[k] if isinstance(bs[k], int) else bs[k].item()
                        losses[k] += v * n
                        samples[k] += n

                        # Compute metrics incrementally
                        t, p = bt[k], bp[k]
                        for mname, fun in self._metrics.items():
                            metric_val = fun(p, t).item()
                            metric_accumulators[k][mname] += metric_val * n
                else:
                    # Single output case
                    v = bl.item()
                    n = bs if isinstance(bs, int) else bs.item()
                    losses["output"] += v * n
                    samples["output"] += n

                    # Compute metrics incrementally
                    for mname, fun in self._metrics.items():
                        metric_val = fun(bp, bt).item()
                        metric_accumulators["output"][mname] += metric_val * n

                total_loss += batch_loss.item()
                batches += 1

        # Log results
        # Compute per-output losses first
        per_output_losses = {}
        for k in losses.keys():
            avg_loss = losses[k] / samples[k]
            per_output_losses[k] = avg_loss
            self._update_logger(f"{step_type}_{k}_loss", avg_loss)

            # Log averaged metrics
            for mname in self._metrics.keys():
                avg_metric = metric_accumulators[k][mname] / samples[k]
                self._update_logger(f"{step_type}_{k}_{mname}", avg_metric)

        # Compute global loss as MEAN of per-output losses for interpretability
        # (Note: optimization still uses sum/weighted sum for backward)
        if per_output_losses:
            epoch_loss = sum(per_output_losses.values()) / len(per_output_losses)
        else:
            epoch_loss = total_loss / batches

        self._update_logger(f"{step_type}_loss", epoch_loss)

    # ==================================================================
    # FIT
    # ==================================================================

    def fit(self, module, x_data, y_data, extra_losses=[]):
        """
        Fit the model to the provided dataset.

        Parameters
        ----------
        module : torch.nn.Module
            Model to train.

        x_data : tensor or dict[str, Tensor]
            Full input dataset.

        y_data : tensor or dict[str, Tensor]
            Full target dataset.

        extra_losses : list[callable]
            Additional zero-argument loss functions.

        Returns
        -------
        module : torch.nn.Module
            Trained model (best weights if restore_best_weights=True)

        history : pandas.DataFrame
            Training history.
        """

        # ---------------- SPLIT DATA ----------------
        with torch.no_grad():
            if isinstance(y_data, dict):
                arr = torch.cat([v for v in y_data.values()], 1)
            else:
                arr = y_data

            # Use torch operations instead of numpy for better performance
            arr = torch.nanmean(arr, 1).cpu().numpy()

        splits = split_data(
            arr,
            {
                "training": 1 - self._validation_split,
                "validation": self._validation_split,
            },
            5,
        )

        train_idx, val_idx = list(splits.values())

        if isinstance(x_data, dict):
            train_x = {k: v[train_idx] for k, v in x_data.items()}
            val_x = {k: v[val_idx] for k, v in x_data.items()}
        else:
            train_x = x_data[train_idx]
            val_x = x_data[val_idx]

        if isinstance(y_data, dict):
            train_y = {k: v[train_idx] for k, v in y_data.items()}
            val_y = {k: v[val_idx] for k, v in y_data.items()}
        else:
            train_y = y_data[train_idx]
            val_y = y_data[val_idx]

        batch_size = self._batch_size or len(train_idx)

        # Auto-tune num_workers if not explicitly set
        num_workers = self._num_workers
        if num_workers is None:
            num_workers = self._compute_optimal_num_workers(len(train_idx))
            if self._verbose in ("full", "minimal"):
                print(f"Auto-tuned num_workers={num_workers} (dataset_size={len(train_idx)})")

        # Optimize DataLoader for CPU training
        dataloader_kwargs = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": False,  # Only useful for GPU
        }

        # Add prefetch_factor and persistent_workers only when using multiprocessing
        if num_workers > 0:
            dataloader_kwargs["persistent_workers"] = True
            dataloader_kwargs["prefetch_factor"] = 2  # Prefetch 2 batches per worker

        train_loader = DataLoader(
            CustomDataset(train_x, train_y),
            shuffle=True,
            **dataloader_kwargs,
        )
        val_loader = DataLoader(
            CustomDataset(val_x, val_y),
            shuffle=False,  # No need to shuffle validation
            **dataloader_kwargs,
        )

        # ---------------- CPU OPTIMIZATION ----------------
        # Set number of threads for CPU operations
        if not torch.cuda.is_available():
            # Use all available CPU cores for intra-op parallelism
            torch.set_num_threads(torch.get_num_threads())
            # Enable flush denormal for better CPU performance
            try:
                torch.set_flush_denormal(True)
            except AttributeError:
                pass  # Not available in all PyTorch versions

        # ---------------- TORCH.COMPILE OPTIMIZATION ----------------
        # Compile model for faster execution (PyTorch 2.0+)
        # Store original module in case compilation fails during execution
        # and for EMA to ensure consistent parameter names
        self._original_module = module
        if self._use_torch_compile and self._is_torch_compile_available():
            try:
                module = torch.compile(module, mode="reduce-overhead")
                if self._verbose in ("full", "minimal"):
                    print("torch.compile enabled (mode=reduce-overhead)")
            except Exception as e:
                # If compilation fails, continue without it
                if self._verbose in ("full", "minimal"):
                    print(f"torch.compile failed ({type(e).__name__}), continuing without it")
        elif self._use_torch_compile and self._verbose in ("full", "minimal"):
            print("torch.compile requested but not available (missing C++ compiler or PyTorch < 2.0)")

        # ---------------- INITIALIZE OPTIMIZER ----------------
        # Use the first learning rate from the list
        optimizer_kwargs_copy = self._optimizer_kwargs.copy()
        optimizer_kwargs_copy["lr"] = self._learning_rates[self._current_lr_index]

        # Enable fused optimizer if available (AdamW, Adam on CUDA)
        if self._use_fused_optimizer and self._optimizer_class in [
            torch.optim.AdamW,
            torch.optim.Adam,
        ]:
            try:
                optimizer_kwargs_copy["fused"] = True
            except Exception:
                pass  # Fused not available on this PyTorch version

        self._optimizer = self._optimizer_class(
            module.parameters(), **optimizer_kwargs_copy
        )

        # ---------------- INITIALIZE EMA ----------------
        # Always initialize EMA with original module to ensure consistent parameter names
        if self._ema_decay is not None:
            self._ema_state = {
                k: v.detach().clone() for k, v in self._original_module.state_dict().items()
            }

        # ---------------- IF UW → ADD UW PARAMS ----------------
        if isinstance(train_y, dict) and self._use_uncertainty_weighting:
            keys = list(train_y.keys())
            device = next(module.parameters()).device
            self._uw_module = UncertaintyWeighting(keys).to(device)

            self._optimizer.add_param_group({"params": self._uw_module.parameters()})

        # ---------------- TRAINING LOOP ----------------
        best_weights = None

        # Start training timer
        self._logger.start_timer()

        for epoch in range(self._epochs):

            self._update_logger("epoch", epoch + 1)

            module.train()

            # On first epoch, catch InductorError from torch.compile and fall back to eager mode
            if epoch == 0 and self._use_torch_compile and not self._compile_failed:
                try:
                    self._step(module, train_loader, "training", extra_losses)
                except Exception as e:
                    # Check if this is an InductorError (torch.compile failed)
                    if "InductorError" in type(e).__name__ or "InductorError" in str(type(e)):
                        self._compile_failed = True
                        if self._verbose in ("full", "minimal"):
                            print(f"torch.compile failed during execution ({type(e).__name__}), falling back to eager mode")

                        # Reset torch dynamo compilation cache
                        if hasattr(torch, "_dynamo"):
                            torch._dynamo.reset()

                        # Switch back to original uncompiled module
                        module = self._original_module

                        # Re-initialize optimizer with uncompiled module
                        optimizer_kwargs_copy = self._optimizer_kwargs.copy()
                        optimizer_kwargs_copy["lr"] = self._learning_rates[self._current_lr_index]
                        if self._use_fused_optimizer and self._optimizer_class in [
                            torch.optim.AdamW,
                            torch.optim.Adam,
                        ]:
                            try:
                                optimizer_kwargs_copy["fused"] = True
                            except Exception:
                                pass
                        self._optimizer = self._optimizer_class(
                            module.parameters(), **optimizer_kwargs_copy
                        )

                        # Re-run the first step without compilation
                        self._step(module, train_loader, "training", extra_losses)
                    else:
                        # Re-raise if it's a different error
                        raise
            else:
                self._step(module, train_loader, "training", extra_losses)

            # Validation with EMA weights if available
            module.eval()
            original_state = None
            if self._ema_decay is not None:
                original_state = self._apply_ema(module)

            with torch.inference_mode():  # Faster than no_grad for inference
                self._step(module, val_loader, "validation")

            # Restore original weights after validation
            if original_state is not None:
                self._restore_original_weights(module, original_state)

            val_loss = self._logger.get_last_value("validation_loss")
            if val_loss is None:
                raise RuntimeError("Validation loss not found in logger")

            lr = self._optimizer.param_groups[0]["lr"]

            # Log learning rate
            self._logger.log_learning_rate(lr)

            # Update early stopping state
            improved = self._logger.update_early_stopping_state(
                val_loss, self._early_stopping_threshold
            )

            if improved:
                if self._restore_best_weights:
                    # Save EMA weights if available, otherwise save current weights
                    if self._ema_state is not None:
                        best_weights = {k: v.cpu().clone() for k, v in self._ema_state.items()}
                    else:
                        best_weights = {k: v.cpu().clone() for k, v in self._original_module.state_dict().items()}

            # Print epoch summary
            self._logger.print_epoch_summary(epoch + 1, val_loss, lr, self._verbose)

            # Early stopping check with learning rate scheduling
            if self._logger.epochs_without_improvement >= self._early_stopping_patience:
                # Check if there are more learning rates to try
                if self._current_lr_index < len(self._learning_rates) - 1:
                    # Move to next learning rate
                    self._current_lr_index += 1
                    next_lr = self._learning_rates[self._current_lr_index]

                    # Restore best weights found so far
                    if best_weights is not None:
                        self._original_module.load_state_dict(best_weights)

                    # Update optimizer learning rate
                    for pg in self._optimizer.param_groups:
                        pg["lr"] = next_lr

                    # Reset early stopping counters
                    self._logger._epochs_without_improvement = 0

                    if self._verbose == "full":
                        print(
                            f"\n[LR Schedule] Switching to learning rate {next_lr:.2e} "
                            f"({self._current_lr_index + 1}/{len(self._learning_rates)})"
                        )
                else:
                    # No more learning rates to try, stop training
                    if self._restore_best_weights and best_weights is not None:
                        self._original_module.load_state_dict(best_weights)
                    break

        print("")
        self._original_module.eval()

        # Always return the original module (not the compiled wrapper)
        return self._original_module, self._logger.to_dataframe()
