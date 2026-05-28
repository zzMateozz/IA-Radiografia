# Fase de Diseño del Sistema

**Proyecto de Aula — Inteligencia Artificial — III Corte (Fase 3)**
Universidad Popular del Cesar — Facultad de Ingeniería de Sistemas — 2026-I

**Integrantes:** Mateo López Patiño, Anaclaudia Vega Martínez, Tonny Enrique Jiménez Márquez

---

## Resumen ejecutivo

Este documento desarrolla el **Objetivo Específico 4.2.2** definido en la Fase 2:

> *Diseñar la arquitectura, el pipeline de datos y el protocolo de evaluación del modelo de inteligencia artificial para la clasificación de radiografías de tórax.*

Cada decisión de esta fase está **respaldada por un hallazgo concreto del EDA** (ver `docs/01_fase_analisis.md`). El resultado es un conjunto de cinco scripts y un notebook listos para ejecutar:

| Componente | Archivo | Función |
|---|---|---|
| Generación de splits | `src/data/build_splits.py` | Construye train/val/test sin data leakage por paciente |
| Preprocesamiento | `src/data/transforms.py` | Pipeline de transformaciones + data augmentation |
| Dataset PyTorch | `src/data/dataset.py` | Clase `ChestXrayDataset` con lazy loading |
| Modelo | `src/models/densenet_chexnet.py` | DenseNet-121 con cabeza multi-etiqueta |
| Entrenamiento | `notebooks/02_kaggle_training.ipynb` | Notebook autocontenido para Kaggle |

---

## 1. Arquitectura general del sistema

```
                +----------------------------+
                |    NIH ChestX-ray14 ZIP    |
                | (45 GB, 112,120 imagenes)  |
                +-------------+--------------+
                              |
                              v
              +-------------------------------+
              | extract_metadata.py           |
              | -> data/raw/Data_Entry_2017.. |
              +---------------+---------------+
                              |
                              v
              +-------------------------------+
              | notebooks/01_eda_metadata     |
              | -> data/processed/            |
              |     metadata_clean.csv        |
              +---------------+---------------+
                              |
                              v
              +-------------------------------+
              | src/data/build_splits.py      |
              | -> metadata_with_splits.csv   |
              |    (train | val | test)       |
              +---------------+---------------+
                              |
       Local                  |                   Kaggle (GPU)
       ----------+-----------+----------------------+----------
                 |                                  |
                 v                                  v
      +----------+--------+              +----------+----------+
      | Inference UI      |              | 02_kaggle_training  |
      | (Streamlit)       | <----------- |   .ipynb            |
      +-------------------+   best_model |                     |
                              .pt        +---------------------+
```

**Razones del diseño**:

- **Separación local / nube**: el EDA y los splits se ejecutan en local (CPU). El entrenamiento requiere GPU CUDA, no disponible en la AMD RX 580 del equipo, así que se delega a Kaggle Notebooks (Tesla T4 / P100 gratuitos).
- **Archivos intermedios livianos**: solo se transfieren a Kaggle el CSV de splits (~6 MB). Las imágenes ya están disponibles allá como dataset oficial `nih-chest-xrays/data`.
- **Reproducibilidad**: cada etapa es un script o notebook con argumentos explícitos y semilla fija.

---

## 2. Pipeline de datos

### 2.1. Generación de splits — `build_splits.py`

| Aspecto | Decisión | Justificación (sección del EDA) |
|---|---|---|
| Test set | Splits oficiales NIH (`test_list.txt`) | Sección 7: ya están bien construidos, sin pacientes compartidos con train_val |
| Train/Val | `GroupKFold(n_splits=5, groups=patient_id)` sobre `train_val_list.txt` | Sección 6: pacientes con hasta 184 imágenes; un split aleatorio causaría leakage |
| Fold de validación | Configurable con `--fold 0..4` | Permite k-fold cross-validation futura sin reescribir el script |
| Verificación | `check_leakage()` confirma intersección 0 entre splits | Cierra la decisión más crítica del EDA |

**Resultado real obtenido al ejecutar el script** (semilla 42, fold 0):

| Split | Imágenes | % | Pacientes |
|---|---|---|---|
| train | 69.219 | 61,74 % | 22.407 |
| val | 17.305 | 15,43 % | 5.601 |
| test | 25.596 | 22,83 % | 2.797 |

`Leakage check: OK (train-val=0, train-test=0, val-test=0)`

### 2.2. Preprocesamiento — `transforms.py`

| Operación | Valor | Justificación |
|---|---|---|
| Resize | 224 × 224 px | Estándar para modelos pre-entrenados en ImageNet (CheXNet usa el mismo tamaño) |
| Conversión a 3 canales | `Grayscale(num_output_channels=3)` | Las radiografías son monocromáticas; el backbone espera RGB |
| Normalización | `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]` | Estadísticas de ImageNet (transfer learning) |
| Rotación | ±7° | La anatomía torácica tiene orientación fija; solo pequeñas variaciones realistas |
| Traslación | ±5% | Compensa pacientes ligeramente descentrados |
| Brillo / contraste | ±10% | Compensa diferencias de exposición entre equipos |
| **Horizontal flip** | **No se aplica** | Cambiaría la lateralidad cardíaca y afectaría Cardiomegaly |
| Augmentation en val/test | Desactivada | Las métricas deben ser deterministas |

### 2.3. Dataset PyTorch — `dataset.py`

- **Lazy loading**: las 112.120 imágenes pesan ≈42 GB; `__getitem__` abre cada PNG bajo demanda.
- **Manejo de errores**: si una imagen falta o está corrupta, devuelve una imagen negra y registra un warning, evitando que el entrenamiento se detenga.
- **Output multi-etiqueta**: cada item devuelve `(imagen_tensor, label_vector_14)` donde `label_vector_14` es un tensor `float32` con valores `{0.0, 1.0}`.
- **Modo `return_meta`**: opcionalmente devuelve `patient_id`, `age`, `gender`, `view` para análisis estratificados durante la evaluación.

### 2.4. Cálculo de `pos_weight`

La función `compute_pos_weight()` computa, para cada patología:

\[ \text{pos\_weight}_i = \frac{\#\text{negativos}_i}{\#\text{positivos}_i} \]

Esto compensa el desbalance extremo (Infiltration : Hernia ≈ 87 : 1, sección 3.1 del EDA). Los valores aproximados sobre el split train son:

| Patología | pos_weight aproximado |
|---|---|
| Hernia | ≈ 626 |
| Pneumonia | ≈ 94 |
| Fibrosis | ≈ 76 |
| Edema | ≈ 58 |
| Cardiomegaly | ≈ 50 |
| ... | ... |
| Infiltration | ≈ 4,6 |

Es decir: penaliza hasta 626 veces más un falso negativo en Hernia que un falso positivo, forzando al modelo a aprender las clases raras.

---

## 3. Diseño del modelo

### 3.1. Arquitectura — `DenseNetCheXNet`

| Componente | Decisión | Justificación |
|---|---|---|
| Backbone | DenseNet-121 pre-entrenado en ImageNet | Estado del arte para NIH-CXR (CheXNet, Rajpurkar et al. 2017); las conexiones densas reutilizan features útil para patologías co-ocurrentes (sección 4.2 del EDA) |
| Cabeza original | `Linear(1024 → 1000)` (ImageNet) | — |
| Cabeza nueva | `Linear(1024 → 14)` con inicialización Xavier | 14 patologías, una neurona por clase |
| Activación de salida | **No aplicada en `forward()`** (logits crudos) | `BCEWithLogitsLoss` aplica sigmoid internamente con estabilidad numérica |
| Dropout antes de la cabeza | Configurable (default 0.0) | Se puede activar si hay overfitting |
| Total de parámetros | ≈ 7,98 millones | Manejable en GPU de Kaggle |

```
Input  (B, 3, 224, 224)  --[norm ImageNet]
   |
   v
DenseNet-121 features
   |
   v
GlobalAveragePool (1024 features)
   |
   v
Linear(1024 -> 14)  --> 14 logits
   |
   v
[Solo en inferencia: sigmoid] --> 14 probabilidades [0..1]
```

### 3.2. Función de pérdida

```python
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

- **¿Por qué BCE y no Cross-Entropy?** Cada clase es independiente. Una imagen puede tener simultáneamente Cardiomegaly, Effusion e Infiltration. BCE evalúa cada salida por separado; Cross-Entropy fuerza a elegir UNA clase.
- **¿Por qué con logits?** `BCEWithLogitsLoss` combina sigmoid + BCE en un solo paso usando log-sum-exp, lo que es numéricamente más estable que aplicar sigmoid manualmente.
- **`pos_weight`** ya descrito en sección 2.4.

### 3.3. Optimizador y scheduler

| Componente | Configuración |
|---|---|
| Optimizador | `Adam(lr=1e-4, weight_decay=1e-5)` |
| Scheduler | `ReduceLROnPlateau(mode='max', factor=0.1, patience=2)` monitoreando AUC en val |
| Mixed Precision | `torch.cuda.amp.GradScaler()` (acelera ~2× en Tesla T4) |
| Batch size | 32 (limitado por memoria de la GPU de Kaggle) |
| Épocas | 10 (configurable) |

Cuando la AUC en validación deja de mejorar 2 épocas seguidas, el LR baja a `1e-5`. Si vuelve a estancarse, `1e-6`. Esto se conoce como *learning rate annealing on plateau* y es lo que usa CheXNet original.

---

## 4. Protocolo de evaluación

### 4.1. Métricas

| Métrica | Por qué se usa |
|---|---|
| **AUC-ROC promedio** (macro) | Métrica principal. Insensible al desbalance entre splits (sección 7.2 del EDA). Permite comparar con CheXNet (~0.84) |
| **AUC-ROC por clase** | Identifica qué patologías aprende mejor el modelo |
| **F1-Score por clase** | Balance entre precisión y recall por patología |
| **Average Precision (AP)** | Equivalente al área bajo la curva precisión-recall, robusta al desbalance |
| **Loss promedio** | Para detectar overfitting comparando train vs val |

**Métricas que NO se usan**:

- **Accuracy global**: engaña ante desbalance extremo. Un modelo que siempre dice "no Hernia" tiene 99,8% de accuracy y es inútil.

### 4.2. Conjuntos de evaluación

- **Validación**: 17.305 imágenes de 5.601 pacientes. Se evalúa al final de cada época. Determina el "mejor checkpoint".
- **Test**: 25.596 imágenes de 2.797 pacientes (split oficial NIH). Solo se usa una vez al final, con el mejor checkpoint, para reportar resultados publicables.

### 4.3. Documentación obligatoria

En el reporte de la Fase 3 se incluirá:

1. **Comparación train vs val vs test**, destacando la caída esperada en test (debido al sesgo de prevalencia documentado en EDA sección 7.2).
2. **AUC por clase** ordenada y comparada con los valores reportados por CheXNet.
3. **Curvas de entrenamiento** (loss y AUC vs época).
4. **Análisis estratificado** por edad, género y vista (usando el modo `return_meta=True` del dataset).

### 4.4. Outputs guardados

| Archivo | Contenido |
|---|---|
| `/kaggle/working/best_model.pt` | Checkpoint con `model_state_dict`, `epoch`, `val_auc`, `config` |
| `/kaggle/working/history.json` | Loss y AUC por época |
| `/kaggle/working/test_results.json` | AUC, F1, AP por clase + AUC promedio |
| `/kaggle/working/test_predictions.npz` | `probs`, `labels`, `image_ids` (para análisis posterior) |
| `/kaggle/working/training_curves.png` | Gráficas de convergencia |

---

## 5. Reproducibilidad

| Mecanismo | Implementación |
|---|---|
| Semilla aleatoria | `--seed 42` en `build_splits.py`; `set_seed(42)` en el notebook |
| Versiones de paquetes | `requirements.txt` con versiones congeladas |
| Splits congelados | `metadata_with_splits.csv` se guarda y se sube a Kaggle como dataset privado |
| Checkpoint | `best_model.pt` incluye el dict de configuración usado |
| Logs | `history.json` registra cada época |

---

## 6. Trazabilidad EDA → Diseño

Tabla maestra que liga cada decisión de diseño con el hallazgo del análisis:

| Hallazgo del EDA | Decisión de diseño |
|---|---|
| Multi-etiqueta: hasta 9 patologías por imagen (§4.1) | 14 sigmoid + BCE en lugar de softmax + CE |
| Desbalance 87× (§3.1) | `BCEWithLogitsLoss(pos_weight=neg/pos)` |
| Co-ocurrencia 30-43% entre patologías (§4.2) | DenseNet-121 (representaciones compartidas) |
| Pacientes con hasta 184 imágenes (§6) | `GroupKFold(groups=patient_id)`; nunca `random_split` |
| Sin leakage en splits NIH (§7.1) | Respetar `train_val_list.txt` y `test_list.txt` |
| Test enriquecido en enfermos (§7.2) | AUC-ROC como métrica principal; documentar caída |
| Resolución 2.646 × 2.486 px (§8.1) | Resize a 224 × 224 (compatibilidad ImageNet) |
| 0 valores nulos (§1.3) | No se requiere imputación |
| Edad como predictor (§5.2) | Modo `return_meta` para análisis estratificado posterior |
| GPU AMD sin CUDA | Kaggle Notebooks (Tesla T4) |

---

## 7. Próximos pasos (Fase de Implementación + Pruebas)

| Tarea | Responsable / archivo |
|---|---|
| Subir `metadata_with_splits.csv` como dataset privado en Kaggle | Equipo |
| Ejecutar `02_kaggle_training.ipynb` con 10 épocas | Kaggle (GPU T4) |
| Descargar `best_model.pt` y `test_results.json` | Equipo |
| Implementar Grad-CAM para interpretabilidad | `src/models/gradcam.py` (pendiente) |
| Construir UI de demostración | `src/app/streamlit_app.py` (pendiente) |
| Validar Grad-CAM contra `BBox_List_2017.csv` (984 cajas) | Equipo |
| Escribir documento de Fase de Pruebas | `docs/03_fase_pruebas.md` |
