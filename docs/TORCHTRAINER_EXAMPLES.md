# TorchTrainer - Esempi Pratici

Questa guida mostra esempi concreti di utilizzo di TorchTrainer con i nuovi defaults ottimizzati.

---

## 🚀 Quick Start - Configurazione Minima

Il modo più semplice per ottenere performance ottimali:

```python
import torch
from labanalysis.modelling.pytorch.utils import (
    TorchTrainer,
    CustomDataset,
    MAEMetric,
    ComboLoss,
    PinballLoss,
    QuantilicRangeLoss,
)

# 1. Crea il tuo modello
class MyModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, output_dim),
        )
    
    def forward(self, x):
        return self.net(x)

# 2. Prepara i dati
x_train = torch.randn(1000, 10)  # 1000 samples, 10 features
y_train = torch.randn(1000, 3)   # 3 outputs

# 3. Crea il trainer con defaults ottimizzati
trainer = TorchTrainer()  # Già ottimizzato per CPU!

# 4. Inizializza e train il modello
model = MyModel(input_dim=10, output_dim=3)
model, history = trainer.fit(model, x_train, y_train)

# 5. Analizza i risultati
import pandas as pd
df_history = pd.DataFrame(history)
print(df_history.tail())
```

**Performance**: ~2.5-3.5x più veloce rispetto alla baseline!

---

## 💾 Memoria Limitata - Gradient Accumulation

Quando la memoria RAM è limitata ma vuoi batch size grandi:

```python
# Sistema con poca RAM
trainer = TorchTrainer(
    batch_size=64,                   # Batch piccolo che entra in memoria
    gradient_accumulation_steps=4,   # Accumula 4 batch prima di update
    # Effective batch size = 64 * 4 = 256
)

model, history = trainer.fit(model, x_train, y_train)
```

**Vantaggi**:
- Memoria: usa solo 64 samples per batch
- Convergenza: come se avessi batch_size=256
- Speedup: performance simile a batch grande

---

## 📊 Massima Stabilità - EMA Weights

Per training lunghi o quando serve massima stabilità:

```python
trainer = TorchTrainer(
    ema_decay=0.999,  # Exponential Moving Average dei pesi
    
    # Altri parametri per stabilità
    gradient_clip_val=1.0,              # Default, già abilitato
    early_stopping_patience=500,        # Più paziente
    early_stopping_threshold=1e-6,      # Soglia più stretta
)

model, history = trainer.fit(model, x_train, y_train)
```

**Quando usare EMA**:
- Training > 1000 epochs
- Validation loss molto variabile
- Modelli che tendono a overfitting
- Quando serve massima affidabilità

**Trade-off**: Raddoppia memoria per i pesi del modello

---

## ⚡ Massima Velocità - CPU Multi-Core

Per sistemi con CPU potenti (6+ cores):

```python
trainer = TorchTrainer(
    # Multiprocessing per data loading
    num_workers=4,                      # 2-4 worker per CPU multi-core
    
    # Batch più grande per sfruttare vectorization
    batch_size=512,                     # Aumenta se hai RAM sufficiente
    
    # Tutti i defaults ottimizzati sono già attivi:
    # - use_torch_compile=True (PyTorch 2.0+)
    # - use_fused_optimizer=True
    # - optimizer_class=AdamW
)

model, history = trainer.fit(model, x_train, y_train)
```

**⚠️ Nota Windows**: `num_workers > 0` può essere instabile. Testa prima con `num_workers=2`, poi aumenta gradualmente.

**Performance attesa**: fino a 4x più veloce su CPU 8+ core.

---

## 🎯 Configurazione Bilanciata - Best Practices

Configurazione raccomandata che bilancia velocità, stabilità e memoria:

```python
trainer = TorchTrainer(
    # Loss personalizzata
    loss=ComboLoss(
        PinballLoss(0.5),           # Quantile regression
        QuantilicRangeLoss(0.99),   # Intervallo di confidenza
    ),
    
    # Metriche
    metrics=[MAEMetric()],
    
    # Optimizer con weight decay
    optimizer_class=torch.optim.AdamW,  # Default
    optimizer_kwargs={
        "lr": [1e-3, 1e-4, 1e-5],      # LR scheduling automatico
        "weight_decay": 1e-5,           # Regularizzazione
    },
    
    # Training configuration
    epochs=50000,
    batch_size=256,                     # Default ottimizzato
    
    # Early stopping
    early_stopping_threshold=1e-5,
    early_stopping_patience=200,
    validation_split=0.2,
    restore_best_weights=True,
    
    # Stabilità
    gradient_clip_val=1.0,              # Default
    ema_decay=None,                     # Opzionale: 0.999 per più stabilità
    
    # Performance
    num_workers=0,                      # Aumenta a 2-4 se sistema stabile
    use_torch_compile=True,             # Default
    use_fused_optimizer=True,           # Default
    
    # Output
    verbose="minimal",
)

model, history = trainer.fit(model, x_train, y_train)
```

---

## 🔄 Multi-Output Models

Quando il modello ha multiple uscite (dict output):

```python
class MultiOutputModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = torch.nn.Linear(10, 64)
        self.output1 = torch.nn.Linear(64, 1)
        self.output2 = torch.nn.Linear(64, 1)
    
    def forward(self, x):
        shared = torch.relu(self.shared(x))
        return {
            "velocity": self.output1(shared),
            "force": self.output2(shared),
        }

# Prepara dati multi-output
x_train = torch.randn(1000, 10)
y_train = {
    "velocity": torch.randn(1000, 1),
    "force": torch.randn(1000, 1),
}

# Trainer con Uncertainty Weighting
trainer = TorchTrainer(
    use_uncertainty_weighting=True,  # Bilancia automaticamente le loss
)

model = MultiOutputModel()
model, history = trainer.fit(model, x_train, y_train)

# Analizza metriche per output
df = pd.DataFrame(history)
print(df[["training_velocity_loss", "validation_velocity_loss"]])
print(df[["training_force_loss", "validation_force_loss"]])
```

---

## 📈 Monitoring e Debugging

### Verbose Output - Full Mode

Per vedere tutti i dettagli durante il training:

```python
trainer = TorchTrainer(
    verbose="full",  # Mostra loss e metriche per ogni output
)

model, history = trainer.fit(model, x_train, y_train)
```

Output esempio:
```
================================================================================
Epoch 100 | LR: 1.00e-03 | Best: 0.123456 | No Improve: 5 | Gap: 195 | Time: 2m 15s
--------------------------------------------------------------------------------
Training Loss:   0.145623
Validation Loss: 0.123456
--------------------------------------------------------------------------------

VELOCITY:
  Loss:    train=0.089234  val=0.067890
  MAE:     train=0.234567  val=0.198765

FORCE:
  Loss:    train=0.056389  val=0.055566
  MAE:     train=0.178234  val=0.165432
================================================================================
```

### Verbose Output - Minimal Mode (Multi-Output)

Quando hai multi-output, `verbose="minimal"` mostra valori aggregati:

```python
trainer = TorchTrainer(
    verbose="minimal",  # Default
)

# Con modello multi-output
model, history = trainer.fit(multi_output_model, x_train, y_train)
```

Output esempio:
```
Epoch 100 | train=0.1456 | val=0.1235 | mae:t=0.2065/v=0.1821 | lr=1.00e-03 | gap=195 | time=2m 15s
```

**Cosa significa ogni valore?**

| Campo | Significato | Calcolo |
|-------|-------------|---------|
| `train=0.0728` | Training loss globale | **MEDIA** delle loss di tutti gli output |
| `val=0.0618` | Validation loss globale | **MEDIA** delle loss di tutti gli output |
| `mae:t=0.2065` | MAE training | **MEDIA** delle MAE di tutti gli output |
| `mae:v=0.1821` | MAE validation | **MEDIA** delle MAE di tutti gli output |

**Esempio con 2 output (velocity, force):**
```python
# Singole loss per output (salvate nell'history ma non mostrate in minimal):
# velocity: train_loss=0.0892, val_loss=0.0679
# force:    train_loss=0.0564, val_loss=0.0556

# Mostrato in minimal:
# train = (0.0892 + 0.0564) / 2 = 0.0728  ← MEDIA
# val   = (0.0679 + 0.0556) / 2 = 0.0618  ← MEDIA

# Singole MAE per output:
# velocity: train_mae=0.2346, val_mae=0.1988
# force:    train_mae=0.1782, val_mae=0.1654

# Mostrato in minimal:
# mae:t = (0.2346 + 0.1782) / 2 = 0.2064  ← MEDIA
# mae:v = (0.1988 + 0.1654) / 2 = 0.1821  ← MEDIA
```

**Importante**: 
- La loss mostrata è la **MEDIA** per interpretabilità e comparabilità tra modelli
- L'ottimizzazione (backward) usa la **SOMMA** (o somma pesata con uncertainty weighting)
- Questo permette di confrontare modelli con diverso numero di output

### Profiling per Bottleneck

Identifica colli di bottiglia nel training:

```python
import torch
from torch.profiler import profile, ProfilerActivity

trainer = TorchTrainer(
    epochs=10,  # Poche epoch per profiling
)

with profile(activities=[ProfilerActivity.CPU]) as prof:
    model, history = trainer.fit(model, x_train, y_train)

# Mostra le operazioni più lente
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
```

---

## 🎓 Tips & Tricks

### 1. Trovare il batch_size ottimale

```python
# Test diversi batch sizes
for bs in [64, 128, 256, 512]:
    trainer = TorchTrainer(
        batch_size=bs,
        epochs=100,  # Poche epoch per test
    )
    import time
    start = time.time()
    model, _ = trainer.fit(MyModel(10, 3), x_train, y_train)
    elapsed = time.time() - start
    print(f"Batch size {bs}: {elapsed:.2f}s")
```

### 2. Salvare i best weights

```python
trainer = TorchTrainer(
    restore_best_weights=True,  # Default
)

model, history = trainer.fit(model, x_train, y_train)

# Salva il modello ottimale
torch.save(model.state_dict(), "best_model.pt")
```

### 3. Resume training con nuovo LR

```python
# Primo training
trainer1 = TorchTrainer(
    optimizer_kwargs={"lr": 1e-3},
    epochs=5000,
)
model, history1 = trainer1.fit(model, x_train, y_train)

# Continua con LR più basso
trainer2 = TorchTrainer(
    optimizer_kwargs={"lr": 1e-4},
    epochs=5000,
)
model, history2 = trainer2.fit(model, x_train, y_train)
```

### 4. Custom Loss Function

```python
class CustomLoss(torch.nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, y_pred, y_true):
        mse = torch.mean((y_pred - y_true) ** 2)
        mae = torch.mean(torch.abs(y_pred - y_true))
        return self.alpha * mse + (1 - self.alpha) * mae

trainer = TorchTrainer(
    loss=CustomLoss(alpha=0.7),
)
```

---

## 📊 Export e Visualizzazione

### Esporta history a CSV

```python
model, history = trainer.fit(model, x_train, y_train)

# Converti a DataFrame e salva
df = pd.DataFrame(history)
df.to_csv("training_history.csv", index=False)
```

### Plot training curves

```python
import plotly.graph_objects as go

df = pd.DataFrame(history)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df["epoch"],
    y=df["training_loss"],
    name="Training Loss",
    mode="lines",
))
fig.add_trace(go.Scatter(
    x=df["epoch"],
    y=df["validation_loss"],
    name="Validation Loss",
    mode="lines",
))
fig.update_layout(
    title="Training History",
    xaxis_title="Epoch",
    yaxis_title="Loss",
)
fig.show()
```

---

## 🔗 Link Utili

- 📚 [CPU Optimization Guide](CPU_OPTIMIZATION_GUIDE.md) - Guida completa alle ottimizzazioni
- 📝 [Changelog](TORCHTRAINER_CHANGELOG.md) - Novità e breaking changes
- 🏠 [Main README](../readme.md) - Documentazione generale labanalysis

---

**Domande?** Contatta il team Technogym Scientific Research
