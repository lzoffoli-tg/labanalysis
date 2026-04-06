# Multi-Output Loss Design - Perché MEDIA invece di SOMMA

Questo documento spiega il design delle loss per modelli multi-output in TorchTrainer.

---

## 🎯 TL;DR

**Ottimizzazione (backward)**: usa SOMMA (o weighted sum)  
**Logging/Monitoring**: usa MEDIA

Questo garantisce:
✅ Corretta ottimizzazione dei gradienti  
✅ Loss interpretabile e comparabile  
✅ Early stopping stabile tra modelli diversi

---

## ❌ Problema con la SOMMA nel logging

### Scenario 1: Modelli con diverso numero di output

```python
# Modello A: 2 output (velocity, force)
model_A = MultiOutputModel(num_outputs=2)
trainer.fit(model_A, x, y)
# Output: val_loss = 0.150 (0.075 + 0.075)

# Modello B: 3 output (velocity, force, power)  
model_B = MultiOutputModel(num_outputs=3)
trainer.fit(model_B, x, y)
# Output: val_loss = 0.225 (0.075 + 0.075 + 0.075)
```

**Problema**: La loss di B è 50% più alta, anche se ogni output ha la stessa qualità!

### Scenario 2: Early Stopping

```python
trainer = TorchTrainer(
    early_stopping_threshold=1e-5,  # Soglia fissa
)

# Con 2 output: threshold OK
# Con 5 output: threshold troppo stretta (loss è 2.5x più alta)
```

**Problema**: Il threshold deve scalare con il numero di output.

### Scenario 3: Comparazione tra Esperimenti

```python
# Esperimento 1: predici solo velocità
val_loss = 0.075

# Esperimento 2: aggiungi predizione forza
val_loss = 0.150  # Raddoppia! 😱

# Esperimento 3: aggiungi anche potenza
val_loss = 0.225  # Triplica! 😱😱
```

**Problema**: Impossibile confrontare esperimenti con diverso numero di output.

---

## ✅ Soluzione: MEDIA per il logging

### Implementazione in TorchTrainer

```python
# _step() method - linee 1345-1362

# 1. Calcola loss per ogni output
for k in losses.keys():
    avg_loss = losses[k] / samples[k]
    per_output_losses[k] = avg_loss
    
# 2. Logga MEDIA per interpretabilità
epoch_loss = sum(per_output_losses.values()) / len(per_output_losses)
self._update_logger(f"{step_type}_loss", epoch_loss)
```

### Backward: continua a usare SOMMA

```python
# _step() method - linee 1281-1287

# Durante training, backward usa SOMMA (corretto!)
if isinstance(bl, dict):
    batch_loss = sum(bl.values())  # O weighted sum con UW
else:
    batch_loss = bl

# batch_loss.backward() ottimizza la somma (giusto!)
```

---

## 📊 Esempi Pratici

### Esempio 1: Comparazione Modelli

```python
# Modello A: 2 output
# velocity: loss=0.080, force: loss=0.070
# Logged loss = (0.080 + 0.070) / 2 = 0.075  ← MEDIA

# Modello B: 3 output  
# velocity: loss=0.080, force: loss=0.070, power: loss=0.075
# Logged loss = (0.080 + 0.070 + 0.075) / 3 = 0.075  ← MEDIA

# Ora sono comparabili! ✅
```

### Esempio 2: Early Stopping

```python
trainer = TorchTrainer(
    early_stopping_threshold=1e-5,  # Funziona per qualsiasi numero di output
)

# Con 2 output: threshold = 1e-5 per output
# Con 5 output: threshold = 1e-5 per output (stesso comportamento)
```

### Esempio 3: Interpretabilità

```python
# Logged: val_loss = 0.075
# Significa: "in media, ogni output ha loss di 0.075"
# Interpretabile e confrontabile! ✅
```

---

## 🔬 Background Teorico

### Perché backward usa SOMMA?

L'ottimizzazione richiede la somma per correttezza matematica:

```
L_total = L_1 + L_2 + ... + L_n
∂L_total/∂θ = ∂L_1/∂θ + ∂L_2/∂θ + ... + ∂L_n/∂θ
```

Dividere per N cambierebbe solo la magnitudine del gradiente (equivale a cambiare learning rate).

### Perché logging usa MEDIA?

Per interpretabilità e comparabilità:

```
L_mean = (L_1 + L_2 + ... + L_n) / n
```

- Normalizzata per numero di output
- Confrontabile tra modelli
- Interpretabile: "loss media per output"

---

## 📝 Best Practices

### 1. Usa loss normalizzate per output

```python
class MyLoss(torch.nn.Module):
    def forward(self, y_pred, y_true):
        # Normalizza per dimensione output
        loss = torch.mean((y_pred - y_true) ** 2)  # Non sum!
        return loss
```

### 2. Threshold di early stopping

```python
# Con MEDIA logging, usa threshold assoluti
trainer = TorchTrainer(
    early_stopping_threshold=1e-5,  # Funziona per qualsiasi N output
)
```

### 3. Uncertainty Weighting

```python
# Con UW, la loss ottimizzata è weighted sum
# Ma la loss loggata resta MEDIA per interpretabilità
trainer = TorchTrainer(
    use_uncertainty_weighting=True,
)
```

---

## 🔄 Migration da SOMMA a MEDIA

Se hai codice esistente che usa SOMMA:

### Cosa cambia?

```python
# Prima (v1.0): SOMMA
# 2 output con loss 0.075 ciascuno
# Logged: val_loss = 0.150

# Dopo (v2.0): MEDIA  
# 2 output con loss 0.075 ciascuno
# Logged: val_loss = 0.075
```

### Early stopping threshold

```python
# Prima (v1.0): threshold scalava con N
old_threshold = 1e-5 * num_outputs

# Dopo (v2.0): threshold assoluto
new_threshold = 1e-5
```

### Confronto con esperimenti precedenti

```python
# Per confrontare con vecchi esperimenti:
old_logged_loss = new_logged_loss * num_outputs
```

---

## 🎓 Riferimenti

### Papers & Resources

- Kendall & Gal (2018): "Multi-Task Learning Using Uncertainty to Weigh Losses" - usa weighted sum per ottimizzazione
- PyTorch Multi-Task Learning: tipicamente usa MEDIA per logging
- Fast.ai: usa MEDIA per comparabilità

### Implementazioni simili

- **TensorFlow/Keras**: `Model.compile(loss=['mse', 'mse'])` → logga MEDIA
- **PyTorch Lightning**: `self.log('val_loss', avg_loss)` → logga MEDIA  
- **Hugging Face Transformers**: logga MEDIA per multiple tasks

---

## ✅ Conclusione

**Design finale in TorchTrainer:**

| Operazione | Aggregazione | Motivo |
|------------|--------------|--------|
| Backward pass | SOMMA (o weighted) | Correttezza matematica |
| Logging | MEDIA | Interpretabilità |
| Early stopping | Usa loss MEDIA | Consistenza tra modelli |
| Metriche | MEDIA | Comparabilità |

Questo design garantisce:
- ✅ Ottimizzazione corretta
- ✅ Loss interpretabile
- ✅ Comparabilità tra esperimenti
- ✅ Early stopping robusto

---

**Domande?** Vedi [TORCHTRAINER_EXAMPLES.md](TORCHTRAINER_EXAMPLES.md) per esempi pratici.
