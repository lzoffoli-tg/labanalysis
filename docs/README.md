# labanalysis - Documentazione

Questa cartella contiene guide e documentazione aggiuntiva per il package `labanalysis`.

## 🆕 Novità v2.0

**TorchTrainer v2.0** introduce ottimizzazioni CPU con speedup 2.5-3.5x! Vedi [V2_SUMMARY.md](V2_SUMMARY.md) per dettagli completi.

**Quick Start:**
```python
trainer = TorchTrainer()  # Defaults già ottimizzati!
model, history = trainer.fit(your_model, x, y)
# 🚀 2.5-3.5x più veloce automaticamente
```

---

## Guide disponibili

### Modelling & Training

- **[CPU_OPTIMIZATION_GUIDE.md](CPU_OPTIMIZATION_GUIDE.md)** - Guida completa alle ottimizzazioni CPU per `TorchTrainer`
  - ⚡ **2.5-3.5x più veloce** con defaults ottimizzati
  - 15 ottimizzazioni implementate (base + avanzate)
  - Configurazioni raccomandate per training su CPU
  - Best practices per batch size, EMA, gradient accumulation
  - Profiling e troubleshooting

- **[TORCHTRAINER_CHANGELOG.md](TORCHTRAINER_CHANGELOG.md)** - Changelog e migration guide
  - Nuovi parametri di default (v2.0)
  - Breaking changes e upgrade guide
  - Feature avanzate (EMA, fused optimizer, gradient accumulation)
  - Migration path per codice esistente

- **[TORCHTRAINER_EXAMPLES.md](TORCHTRAINER_EXAMPLES.md)** - Esempi pratici e use cases
  - 🚀 Quick start con configurazione minima
  - Esempi per memoria limitata, massima stabilità, massima velocità
  - Multi-output models e uncertainty weighting
  - Tips & tricks per tuning e debugging

- **[MULTI_OUTPUT_LOSS_DESIGN.md](MULTI_OUTPUT_LOSS_DESIGN.md)** - Design rationale per multi-output loss
  - Perché MEDIA invece di SOMMA per il logging
  - Background teorico e best practices
  - Esempi pratici e migration guide
  - Comparazione con altre librerie (TensorFlow, PyTorch Lightning)

- **[TESTING_GUIDE.md](TESTING_GUIDE.md)** - Guida ai test per TorchTrainer
  - 🧪 Test aggiornati per v2.0
  - Nuovi test per EMA, gradient accumulation, fused optimizer
  - Come eseguire i test e coverage
  - Troubleshooting e best practices

- **[WINDOWS_COMPILER_SETUP.md](WINDOWS_COMPILER_SETUP.md)** - 🔧 Setup compiler C++ su Windows
  - Guida installazione MSVC per torch.compile()
  - Opzioni: Chocolatey, Build Tools, Visual Studio
  - Verifica installazione e troubleshooting
  - Performance gain: +50-100% speedup

- **[V2_SUMMARY.md](V2_SUMMARY.md)** - ⭐ Riepilogo completo v2.0
  - Tutte le modifiche in un unico documento
  - Performance metrics e breakdown
  - File modificati/creati
  - Migration guide completa
  - Checklist implementazione

## Struttura del package

```
labanalysis/
├── modelling/
│   └── pytorch/
│       └── utils.py          # TorchTrainer, CustomDataset, Loss functions
├── gaitanalysis/             # Analisi del cammino
├── utils/                    # Utility generali
└── docs/                     # Questa cartella
```

## Riferimenti rapidi

### TorchTrainer - Configurazione ottimale CPU

```python
from labanalysis.modelling.pytorch.utils import TorchTrainer

trainer = TorchTrainer(
    batch_size=256,
    num_workers=4,
    use_torch_compile=True,
    verbose="minimal",
)
```

### Link utili

- Repository principale: [labanalysis](../)
- Issues & feedback: Contatta il team di Scientific Research
