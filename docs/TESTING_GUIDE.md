# Testing Guide - TorchTrainer v2.0

Guida ai test per TorchTrainer dopo le modifiche della versione 2.0.

---

## 🧪 Test Aggiornati

### Modifiche ai Test Esistenti

#### 1. **test_init_default** - Nuovi defaults
```python
def test_init_default(self):
    """Test initialization with defaults (v2.0 optimized)."""
    trainer = laban.TorchTrainer()
    
    # Nuovi defaults v2.0:
    assert trainer._batch_size == 256  # Era: None
    assert trainer._gradient_clip_val == 1.0  # Era: None
    assert trainer._use_torch_compile is True  # Nuovo
    assert trainer._use_fused_optimizer is True  # Nuovo
    assert trainer._optimizer_class == torch.optim.AdamW  # Era: Adam
```

#### 2. **test_multi_output_loss_aggregation** - MEDIA invece di SOMMA
```python
def test_multi_output_loss_aggregation(self):
    """Test that multi-output loss uses MEAN aggregation (v2.0)."""
    # ...training...
    
    # Verifica che global_loss = MEDIA(output1_loss, output2_loss)
    global_loss = row["training_loss"]
    expected_mean = (output1_loss + output2_loss) / 2
    assert abs(global_loss - expected_mean) < 1e-5
```

#### 3. **test_print_epoch_summary_minimal_multi_output** - Logger ottimizzato
```python
def test_print_epoch_summary_minimal_multi_output(self):
    """Test that minimal mode shows AVERAGE metrics."""
    # Multi-output metrics dovrebbero essere mediate
    logger.update("training_output1_mae", 0.20)
    logger.update("training_output2_mae", 0.10)
    
    # Output dovrebbe mostrare MAE = (0.20 + 0.10) / 2 = 0.15
```

---

## 🆕 Nuovi Test per Feature v2.0

### TestTorchTrainerV2Features

#### 1. EMA (Exponential Moving Average)
```python
def test_ema_weights(self, simple_model, simple_tensors):
    """Test Exponential Moving Average weights feature."""
    trainer = laban.TorchTrainer(
        ema_decay=0.999,  # Enable EMA
    )
    
    model, history = trainer.fit(simple_model, x, y)
    
    # EMA state dovrebbe essere inizializzato
    assert trainer._ema_state is not None
```

#### 2. Gradient Accumulation
```python
def test_gradient_accumulation(self, simple_model, simple_tensors):
    """Test gradient accumulation feature."""
    trainer = laban.TorchTrainer(
        batch_size=16,
        gradient_accumulation_steps=4,  # Effective batch = 64
    )
    
    model, history = trainer.fit(simple_model, x, y)
    assert len(history) > 0
```

#### 3. Gradient Clipping (Default Enabled)
```python
def test_gradient_clipping_default(self):
    """Test that gradient clipping is enabled by default."""
    trainer = laban.TorchTrainer()
    assert trainer._gradient_clip_val == 1.0  # Default
```

#### 4. Fused Optimizer
```python
def test_fused_optimizer_enabled(self):
    """Test that fused optimizer is attempted."""
    trainer = laban.TorchTrainer(
        optimizer_class=torch.optim.AdamW,
        use_fused_optimizer=True,  # Default
    )
    # Non dovrebbe generare errori
```

#### 5. Default Parameters v2.0
```python
def test_default_optimizer_is_adamw(self):
    """Test that default optimizer is AdamW (v2.0 change)."""
    trainer = laban.TorchTrainer()
    assert trainer._optimizer_class == torch.optim.AdamW

def test_default_batch_size_is_256(self):
    """Test that default batch size is 256 (v2.0 change)."""
    trainer = laban.TorchTrainer()
    assert trainer._batch_size == 256
```

#### 6. EMA con LR Scheduling
```python
def test_ema_with_lr_scheduling(self):
    """Test EMA with learning rate scheduling."""
    trainer = laban.TorchTrainer(
        optimizer_kwargs={"lr": [0.01, 0.001]},
        ema_decay=0.999,
    )
    
    model, history = trainer.fit(simple_model, x, y)
    assert trainer._ema_state is not None
```

---

## 🏃 Eseguire i Test

### Setup Ambiente

Prima di eseguire i test, assicurati di essere nel virtual environment corretto:

```bash
# Attiva il virtual environment
# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate

# Verifica dipendenze
pip install torch pytest pytest-cov
```

### Test Completi
```bash
# Esegui tutti i test
pytest test/test_pytorch_utils.py -v

# Esegui solo i test di TorchTrainer
pytest test/test_pytorch_utils.py::TestTorchTrainer -v

# Esegui solo i nuovi test v2.0
pytest test/test_pytorch_utils.py::TestTorchTrainerV2Features -v
```

### Test Specifici
```bash
# Test defaults
pytest test/test_pytorch_utils.py::TestTorchTrainer::test_init_default -v

# Test multi-output aggregation
pytest test/test_pytorch_utils.py::TestIntegration::test_multi_output_loss_aggregation -v

# Test EMA
pytest test/test_pytorch_utils.py::TestTorchTrainerV2Features::test_ema_weights -v
```

### Con Coverage
```bash
# Coverage report
pytest test/test_pytorch_utils.py --cov=src.labanalysis.modelling.pytorch.utils --cov-report=html

# Apri report
# Windows
start htmlcov/index.html

# Linux/Mac
open htmlcov/index.html
```

---

## 📊 Test Coverage

### Aree Coperte

| Feature | Test | Status |
|---------|------|--------|
| Defaults v2.0 | ✅ `test_init_default` | Pass |
| Multi-output MEAN | ✅ `test_multi_output_loss_aggregation` | Pass |
| EMA | ✅ `test_ema_weights` | Pass |
| Gradient Accumulation | ✅ `test_gradient_accumulation` | Pass |
| Gradient Clipping | ✅ `test_gradient_clipping_default` | Pass |
| Fused Optimizer | ✅ `test_fused_optimizer_enabled` | Pass |
| Logger ottimizzato | ✅ `test_print_epoch_summary_minimal_multi_output` | Pass |
| LR Scheduling + EMA | ✅ `test_ema_with_lr_scheduling` | Pass |

### Coverage Atteso
- **Linee coperte**: ~95%
- **Branch coverage**: ~90%
- **Function coverage**: 100%

---

## 🐛 Troubleshooting

### Test Falliscono su Windows

**Problema**: `num_workers > 0` può essere instabile su Windows.

**Soluzione**: Test usano `num_workers=0` di default (safe).

```python
# Nei test
trainer = laban.TorchTrainer(
    num_workers=0,  # Safe per Windows
)
```

### torch.compile non disponibile

**Problema**: PyTorch < 2.0 non ha `torch.compile()`.

**Soluzione**: Il trainer ha fallback automatico.

```python
# Nel trainer
try:
    module = torch.compile(module)
except Exception:
    pass  # Continua senza compile
```

### Fused Optimizer non supportato

**Problema**: Fused optimizer richiede PyTorch recente.

**Soluzione**: Il trainer tenta ma non fallisce se non disponibile.

```python
# Nel trainer
try:
    optimizer_kwargs_copy["fused"] = True
except Exception:
    pass  # Continua senza fused
```

---

## 📝 Checklist Pre-Commit

Prima di committare modifiche a TorchTrainer:

- [ ] Eseguiti tutti i test: `pytest test/test_pytorch_utils.py -v`
- [ ] Coverage > 90%: `pytest --cov`
- [ ] Test multi-output aggregation passa
- [ ] Test EMA passa
- [ ] Test gradient accumulation passa
- [ ] Documentazione aggiornata
- [ ] Changelog aggiornato

---

## 🔄 Migration Testing

Se hai test esistenti che usano TorchTrainer:

### 1. Verifica i defaults
```python
# Prima (potrebbe fallire)
trainer = TorchTrainer()
# batch_size era None

# Dopo (aggiorna aspettative)
trainer = TorchTrainer()
assert trainer._batch_size == 256  # Nuovo default
```

### 2. Verifica multi-output loss
```python
# Prima
# global_loss = sum(output_losses)

# Dopo
# global_loss = mean(output_losses)
expected_mean = sum(output_losses) / len(output_losses)
assert abs(global_loss - expected_mean) < 1e-5
```

### 3. Test con fixture
```python
@pytest.fixture
def legacy_trainer():
    """Trainer con vecchi defaults per backward compatibility."""
    return laban.TorchTrainer(
        batch_size=None,
        gradient_clip_val=None,
        use_torch_compile=False,
    )
```

---

## 📚 Riferimenti

- [test_pytorch_utils.py](../test/test_pytorch_utils.py) - File test completo
- [TORCHTRAINER_CHANGELOG.md](TORCHTRAINER_CHANGELOG.md) - Modifiche v2.0
- [CPU_OPTIMIZATION_GUIDE.md](CPU_OPTIMIZATION_GUIDE.md) - Guida ottimizzazioni
- [MULTI_OUTPUT_LOSS_DESIGN.md](MULTI_OUTPUT_LOSS_DESIGN.md) - Design MEDIA vs SOMMA

---

## 🎓 Best Practices

### Scrivere nuovi test

1. **Usa fixtures** per modelli e dati comuni
2. **Test isolati**: ogni test dovrebbe essere indipendente
3. **Seed random**: usa `torch.manual_seed()` per riproducibilità
4. **Verbose off**: usa `verbose="off"` nei test per output pulito
5. **Assertions chiare**: messaggi esplicativi per failures

### Esempio test ben scritto

```python
def test_new_feature(self, simple_model, simple_tensors):
    """Test description: what and why."""
    # Setup
    torch.manual_seed(42)  # Reproducibility
    x, y = simple_tensors
    
    # Create trainer with specific config
    trainer = laban.TorchTrainer(
        new_param=value,
        verbose="off",  # Clean output
    )
    
    # Execute
    model, history = trainer.fit(simple_model, x, y)
    
    # Assert with clear messages
    assert condition, f"Expected X but got Y: {detail}"
```

---

**Questions?** Vedi [main README](../readme.md) per documentazione completa.
