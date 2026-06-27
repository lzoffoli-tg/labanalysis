# WholeBody Quick Reference

Guida rapida all'interpretazione di angoli e segmenti del modello biomeccanico WholeBody.

## Indice

- [Convenzioni di Segno](#convenzioni-di-segno)
- [Angoli Pelvi](#angoli-pelvi)
- [Angoli Spalle](#angoli-spalle)
- [Angoli Arti Inferiori](#angoli-arti-inferiori)
- [Angoli Arti Superiori](#angoli-arti-superiori)
- [Angoli Tronco e Collo](#angoli-tronco-e-collo)
- [Segmenti Corporei](#segmenti-corporei)
- [Global vs Local](#global-vs-local)
- [Range Fisiologici](#range-fisiologici)

## Convenzioni di Segno

### Sistema di Riferimento Globale

```
Y (Vertical) ↑
            |
            |
            |______ Z (Anteroposterior)
           /
          /
         X (Lateral)
```

- **X-axis**: Laterale (negativo = destra, positivo = sinistra)
- **Y-axis**: Verticale (negativo = basso, positivo = alto)
- **Z-axis**: Anteroposteriore (negativo = posteriore, positivo = anteriore)

### Regola Generale

| Movimento | Positivo (+) | Negativo (−) |
|-----------|--------------|--------------|
| **Sagittale** | Flessione | Estensione |
| **Frontale** | Abduzione / Lato sinistro elevato | Adduzione / Lato destro elevato |
| **Trasverso** | Rotazione interna / Sinistra avanti | Rotazione esterna / Destra avanti |

## Angoli Pelvi

### Lateral Tilt (Inclinazione Laterale)

```python
# Global: rispetto alla gravità
lateral_global = body.pelvis_lateraltilt_global

# Local: rispetto all'asse del tronco
lateral_local = body.pelvis_lateraltilt_local
```

**Interpretazione**:
- **+10°**: Anca sinistra 10° più alta della destra
- **−10°**: Anca destra 10° più alta della sinistra
- **0°**: Pelvi livellata

**Differenza Global vs Local**:
- Piccola (<5°): Postura eretta, minimo lean laterale del tronco
- Grande (>10°): Compensazione significativa, tronco inclinato lateralmente

**Applicazioni Cliniche**:
- Valutazione dismetria arti inferiori
- Trendelenburg test
- Analisi pattern di compensazione posturale
- Scoliosi funzionale vs strutturale

### Rotation (Rotazione Assiale)

```python
# Global: rispetto alla direzione anteriore globale
rotation_global = body.pelvis_rotation_global

# Local: rispetto al piano trasverso del collo
rotation_local = body.pelvis_rotation_local
```

**Interpretazione**:
- **+15°**: Anca sinistra 15° più avanti della destra (rotazione sinistra)
- **−15°**: Anca destra 15° più avanti della sinistra (rotazione destra)
- **0°**: Anche allineate frontalmente

**Applicazioni Cliniche**:
- Analisi del cammino (dissociazione tronco-pelvi)
- Valutazione mobilità rotazionale dell'anca
- Pattern di movimento durante squat/lunge

### Anterior/Posterior Tilt (Anteroversione/Retroversione)

```python
tilt = body.pelvis_anteroposterior_tilt_global
```

**Interpretazione**:
- **−20°**: Anteroversione 20° (lordosi lombare accentuata)
- **+20°**: Retroversione 20° (schiena piatta)
- **0°**: Neutro

**Range Normale**: −10° a +10°

## Angoli Spalle

### Lateral Tilt (Elevazione Spalle)

```python
# Global: rispetto all'orizzontale
shoulder_global = body.shoulder_lateraltilt_global

# Local: rispetto all'asse del tronco
shoulder_local = body.shoulder_lateraltilt_local
```

**Interpretazione**:
- **+8°**: Spalla sinistra 8° più alta della destra
- **−8°**: Spalla destra 8° più alta della sinistra
- **0°**: Spalle livellate

**Differenza Global vs Local**:
- Rivela pattern di compensazione posturale
- Local isola asimmetrie vere delle spalle dal lean del tronco

**Applicazioni Cliniche**:
- Valutazione postura scapolare
- Asimmetrie funzionali (dominanza mano)
- Scoliosi toracica

### Scapular Protraction/Retraction

```python
left_scap = body.left_scapular_protractionretraction
right_scap = body.right_scapular_protractionretraction
```

**Interpretazione**:
- **+20°**: Protrazione 20° (spalle in avanti, "spalle chiuse")
- **−20°**: Retrazione 20° (scapole addotte, "spalle aperte")

**Applicazioni Cliniche**:
- Upper crossed syndrome
- Postura da computer
- Stabilità scapolare

## Angoli Arti Inferiori

### Anca (Hip)

| Proprietà | Positivo | Negativo | Range Normale |
|-----------|----------|----------|---------------|
| `left_hip_flexionextension` | Flessione | Estensione | −20° a 120° |
| `left_hip_abductionadduction` | Abduzione | Adduzione | −20° a 45° |
| `left_hip_internalexternalrotation` | Rotazione interna | Rotazione esterna | −45° a 45° |

**Note**: Gli angoli sono ~0° quando la coscia è verticale (posizione eretta neutrale)

**Esempi**:
```python
# Durante squat
hip_flex = body.left_hip_flexionextension.max()
if hip_flex > 100:
    print("Squat profondo (>100°)")
elif hip_flex > 90:
    print("Squat parallelo (90-100°)")
else:
    print("Squat parziale (<90°)")
```

### Ginocchio (Knee)

| Proprietà | Positivo | Negativo | Range Normale |
|-----------|----------|----------|---------------|
| `left_knee_flexionextension` | Flessione | Estensione | 0° a 140° |
| `left_knee_varusvalgus` | Varo (gambe arcuate) | Valgo (ginocchia a X) | −5° a +5° |

**Note**: Convenzione segno varo/valgo - **Positivo = Varo**, **Negativo = Valgo**

**Esempi**:
```python
# Valutazione allineamento
varus = body.left_knee_varusvalgus.mean()
if varus > 5:
    print(f"Varo {varus:.1f}° (gambe arcuate)")
elif varus < -5:
    print(f"Valgo {abs(varus):.1f}° (ginocchia a X)")
else:
    print("Allineamento neutro")
```

### Caviglia (Ankle)

| Proprietà | Positivo | Negativo | Range Normale |
|-----------|----------|----------|---------------|
| `left_ankle_flexionextension` | Dorsiflessione | Plantarflessione | −20° a +30° |
| `left_ankle_inversioneversion` | Eversione | Inversione | −30° a +20° |

## Angoli Arti Superiori

### Spalla (Shoulder Joint)

| Proprietà | Positivo | Negativo | Range Normale |
|-----------|----------|----------|---------------|
| `left_shoulder_flexionextension` | Flessione | Estensione | −60° a 180° |
| `left_shoulder_abductionadduction` | Abduzione | Adduzione | 0° a 180° |
| `left_shoulder_internalexternalrotation` | Rotazione interna | Rotazione esterna | −90° a 90° |

**Note**: Gli angoli sono ~0° quando il braccio pende verticalmente al fianco

### Gomito (Elbow)

| Proprietà | Positivo | Negativo | Range Normale |
|-----------|----------|----------|---------------|
| `left_elbow_flexionextension` | Flessione | Estensione | 0° a 150° |

## Angoli Tronco e Collo

### Tronco (Trunk)

```python
# Flessione sagittale (global)
trunk_flex = body.trunk_flexionextension_global

# Flessione laterale (nel sistema pelvi)
trunk_lateral = body.trunk_lateralflexion

# Rotazione (global)
trunk_rot_global = body.trunk_rotation_global

# Rotazione (rispetto alla pelvi)
trunk_rot_local = body.trunk_rotation_local
```

**Interpretazione Rotazione**:
- `trunk_rotation_global`: Rotazione assoluta del tronco nello spazio
- `trunk_rotation_local`: Dissociazione tronco-pelvi (coordinazione)
- Durante il cammino normale: 5-10° di dissociazione

### Collo (Neck)

```python
neck_flex = body.neck_flexionextension
neck_lateral = body.neck_lateralflexion
```

**Interpretazione**:
- Flessione >15°: Forward head posture
- Estensione >10°: Testa retratta

### Curvature Spinali

```python
lordosis = body.lumbar_lordosis    # Lordosi lombare
kyphosis = body.dorsal_kyphosis    # Cifosi toracica
```

**IMPORTANTE**: Questi sono **angoli interni** ai vertici vertebrali!
- **Angoli più piccoli** = curvatura **maggiore**
- **Angoli più grandi** = curvatura **minore** (schiena più piatta)

**Interpretazione**:

| Angolo Lordosi | Interpretazione |
|----------------|-----------------|
| <140° | Iperlordosi (curvatura eccessiva, sway back) |
| 140-160° | Normale |
| >160° | Ipolordosi (schiena piatta lombare) |

| Angolo Cifosi | Interpretazione |
|---------------|-----------------|
| <140° | Ipercifosi (gobba, hunchback) |
| 140-160° | Normale |
| >160° | Ipocifosi (schiena piatta toracica) |

## Segmenti Corporei

### Lunghezze Segmentali

```python
# Accesso aggregato
lengths = body.segment_lengths

# Accesso individuale
thigh_length = body.left_thigh_length
shank_length = body.left_shank_length
foot_length = body.left_foot_length
```

**Utilità**:
- Normalizzazione step length
- Calcoli antropometrici
- Validazione dati (lunghezze costanti)

### Centri Articolari

```python
hip_center = body.left_hip         # De Leva 1996 regression
knee_center = body.left_knee       # Midpoint epicondili
ankle_center = body.left_ankle     # Midpoint malleoli
shoulder_center = body.left_shoulder
```

**Note**: I centri articolari sono **calcolati**, non marcatori fisici

### Reference Frames (Sistemi di Riferimento)

```python
pelvis_rf = body.pelvis_referenceframe
neck_rf = body.neck_referenceframe
```

**Utilità**:
- Trasformazioni di coordinate
- Calcoli angoli local
- Analisi movimenti complessi

## Global vs Local

### Quando Usare Global

**Misure Global** (rispetto alla gravità/terra):
- Valutazione postura assoluta
- Analisi stabilità ed equilibrio
- Performance in compiti funzionali

**Esempio**: Quanto è inclinato il bacino rispetto al pavimento?

```python
pelvis_tilt_global = body.pelvis_lateraltilt_global
```

### Quando Usare Local

**Misure Local** (rispetto a segmento corporeo):
- Isolamento disfunzioni articolari
- Valutazione allineamento segmentale
- Pattern di compensazione

**Esempio**: Il bacino è disallineato rispetto alla colonna, indipendentemente da come il soggetto si sta inclinando?

```python
pelvis_tilt_local = body.pelvis_lateraltilt_local
```

### Interpretazione Differenze

```python
global_angle = body.pelvis_lateraltilt_global.mean()
local_angle = body.pelvis_lateraltilt_local.mean()
difference = abs(global_angle - local_angle)

if difference < 5:
    print("Postura eretta, minima compensazione")
elif difference < 10:
    print("Moderata flessione/inclinazione tronco")
else:
    print("Significativa compensazione posturale")
```

## Range Fisiologici

### Tabella Riassuntiva

| Articolazione | Movimento | Range Normale | Funzionale Minimo |
|---------------|-----------|---------------|-------------------|
| **Anca** | Flessione | 0-120° | 90° (sedersi) |
| | Estensione | 0-20° | 10° (cammino) |
| | Abduzione | 0-45° | 20° |
| **Ginocchio** | Flessione | 0-140° | 90° (scale) |
| **Caviglia** | Dorsiflessione | 0-30° | 10° (cammino) |
| | Plantarflessione | 0-50° | 20° (cammino) |
| **Spalla** | Flessione | 0-180° | 90° (reach) |
| | Abduzione | 0-180° | 90° (reach lateral) |
| **Gomito** | Flessione | 0-150° | 90° (self-care) |
| **Pelvi** | Tilt AP | ±10° | - |
| | Tilt laterale | ±5° | - |
| | Rotazione | ±10° | - |

### Asimmetrie Bilaterali

```python
left = body.left_knee_flexionextension.max()
right = body.right_knee_flexionextension.max()
asymmetry = abs(left - right) / ((left + right) / 2) * 100

if asymmetry < 5:
    print("Simmetria eccellente")
elif asymmetry < 10:
    print("Simmetria accettabile")
else:
    print(f"Asimmetria significativa: {asymmetry:.1f}%")
```

**Soglie Cliniche**:
- <5%: Fisiologico
- 5-10%: Borderline (monitorare)
- >10%: Significativo (investigare)

## Esempi Pratici

### Analisi Squat

```python
# Profondità squat
hip_flex = body.left_hip_flexionextension.max()
knee_flex = body.left_knee_flexionextension.max()

print(f"Hip flexion: {hip_flex:.1f}°")
print(f"Knee flexion: {knee_flex:.1f}°")

# Classificazione
if knee_flex > 120:
    print("Deep squat (full ROM)")
elif knee_flex > 90:
    print("Parallel squat")
else:
    print("Partial squat")

# Varo/valgo dinamico
varus_max = body.left_knee_varusvalgus.max()
varus_min = body.left_knee_varusvalgus.min()
varus_excursion = varus_max - varus_min

if varus_excursion > 10:
    print(f"Warning: Excessive knee varus/valgus excursion ({varus_excursion:.1f}°)")
```

### Analisi Cammino

```python
# ROM caviglia durante ciclo del passo
ankle_rom = (body.left_ankle_flexionextension.max() - 
             body.left_ankle_flexionextension.min())

print(f"Ankle ROM during gait: {ankle_rom:.1f}°")

# Normale: 25-30°
if ankle_rom < 20:
    print("Reduced ankle ROM (stiff gait)")
elif ankle_rom > 35:
    print("Excessive ankle ROM (hypermobility?)")

# Dissociazione tronco-pelvi
dissociation = body.trunk_rotation_local
max_diss = np.abs(dissociation.data).max()

print(f"Max trunk-pelvis dissociation: {max_diss:.1f}°")

# Normale: 5-10°
if max_diss < 5:
    print("Reduced dissociation (rigid gait)")
elif max_diss > 15:
    print("Excessive dissociation")
```

### Valutazione Posturale

```python
# Postura sagittale
pelvic_tilt = body.pelvis_anteroposterior_tilt_global.mean()
lumbar_lordosis = body.lumbar_lordosis.mean()

if pelvic_tilt < -15 and lumbar_lordosis < 140:
    print("Lower crossed syndrome (anterior pelvic tilt + hyperlordosis)")
elif pelvic_tilt > 15 and lumbar_lordosis > 160:
    print("Flat back posture (posterior tilt + reduced lordosis)")

# Postura frontale
pelvis_lateral = body.pelvis_lateraltilt_global.mean()
shoulder_lateral = body.shoulder_lateraltilt_global.mean()

if abs(pelvis_lateral) > 5 or abs(shoulder_lateral) > 5:
    print("Lateral postural asymmetry detected")
    print(f"  Pelvis: {pelvis_lateral:.1f}°")
    print(f"  Shoulders: {shoulder_lateral:.1f}°")
```

## Vedi Anche

- [Joint Angles](joint-angles.md) - Guida dettagliata agli angoli articolari
- [WholeBody Model](whole-body-model.md) - Documentazione completa del modello
- [Coordinate Systems](coordinate-systems.md) - Sistemi di riferimento e trasformazioni
- [API Reference: WholeBody](../../api/records/bodies.md) - API completa

---

**Quick Reference**: Interpretazione rapida angoli e segmenti WholeBody per analisi biomeccanica clinica.
