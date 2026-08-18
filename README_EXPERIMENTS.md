# YOLOv12-Large sobre VisDrone (Windows) — Documentación de experimentos

Documentación independiente del flujo de entrenamiento local en Windows para
`yolo12l.pt` sobre un dataset VisDrone de clase única (`person`), con tracking
en Weights & Biases. No modifica `requirements.txt` ni ningún archivo del
paquete `ultralytics/` original.

## 1. Hardware y entorno

| Componente        | Valor                              |
|--------------------|-------------------------------------|
| SO                 | Windows 11 Pro                      |
| GPU                | NVIDIA GeForce RTX 5060 Ti — 16 GB VRAM |
| CUDA Toolkit       | 13.3 (driver del sistema)           |
| Framework          | PyTorch + CUDA (wheel, ver §2)      |
| Modelo             | YOLOv12-Large (`yolo12l.pt`)        |

> **Nota sobre CUDA 13.3 y wheels de PyTorch**: los wheels oficiales de PyTorch
> se distribuyen con etiquetas `cuXXX` (p. ej. `cu124`, `cu128`) que empaquetan
> su propio runtime CUDA; no necesitan coincidir exactamente con la versión del
> CUDA Toolkit del sistema, solo requieren un **driver NVIDIA igual o más nuevo**
> que el mínimo exigido por ese runtime. La RTX 5060 Ti es arquitectura
> **Blackwell (compute capability `sm_120`)**: si `torch.cuda.is_available()`
> devuelve `True` pero el entrenamiento falla con `no kernel image is
> available for execution`, el wheel instalado es demasiado antiguo para
> Blackwell — reinstala con el canal `cu128` o nightly indicado en §2.

## 2. Diferencias: `requirements.txt` (original) vs `requirements-windows.txt` (nuevo)

| Paquete | `requirements.txt` (original) | `requirements-windows.txt` (Windows) | Motivo |
|---|---|---|---|
| `torch` / `torchvision` | Pineado (`torch==2.2.2`) | **Excluido del archivo** — se instala aparte | Necesita un build reciente con soporte `sm_120` (Blackwell) y el índice CUDA correcto; pinear una versión antigua rompería la RTX 5060 Ti |
| `flash_attn` | Wheel Linux `cp311-linux_x86_64` | **Eliminado** | No existe wheel oficial para Windows; no es necesario, YOLOv12 usa `torch.nn.functional.scaled_dot_product_attention` como fallback |
| `onnxruntime` (CPU) | Incluido junto a `onnxruntime-gpu` | **Eliminado** | Evita conflicto de paquete duplicado (CPU vs GPU) en Windows |
| `onnx` | `1.14.0` | `1.16.1` | Compatibilidad con export en Windows/Python recientes |
| `python-dotenv` | No estaba | **Añadido** | Carga segura de `WANDB_API_KEY` desde `.env` |
| `wandb` | No estaba | **Añadido** | Tracking de métricas |
| Resto (`timm`, `albumentations`, `pycocotools`, `opencv-python`, etc.) | igual | igual | Sin cambios, multiplataforma |

## 3. Instalación manual en Windows (ningún script automático)

Ejecutar en PowerShell, dentro de un entorno virtual (`venv` o `conda`) ya activado:

```powershell
# 1) Actualizar pip
python -m pip install --upgrade pip

# 2) Instalar PyTorch con soporte CUDA (recomendado para RTX 5060 Ti / Blackwell)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 2b) SOLO SI la verificación de GPU falla o aparece "no kernel image is
#     available" (Blackwell aún no soportado en el build estable cu124):
# pip uninstall -y torch torchvision torchaudio
# pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# 3) Instalar el resto de dependencias (archivo independiente, sin tocar requirements.txt)
pip install -r requirements-windows.txt

# 4) Instalar este repo (YOLOv12) en modo editable
pip install -e .
```

> Evita fijar `torch==2.4.0` en Windows: es una versión con errores conocidos
> en CPU/Windows (ver `pyproject.toml:73`). Usa la última estable de la serie
> `2.5.x`/`2.6.x`/`2.7.x`, o el nightly `cu128` indicado arriba si tu GPU lo requiere.

## 4. Estructura de directorios esperada

Ultralytics resuelve el campo `path` de cada `.yaml` de dataset como
`(datasets_dir / path).resolve()`, donde `datasets_dir` es la carpeta
configurada en `ultralytics/settings.json` (por defecto, la carpeta hermana
`datasets/` junto al repo clonado). Estructura esperada:

```
GitHub/
├── YOLOv12/                          ← este repo clonado
│   ├── requirements.txt              (original, sin tocar)
│   ├── requirements-windows.txt      (nuevo)
│   ├── train_yolo12.py               (nuevo)
│   ├── README_EXPERIMENTS.md         (este archivo)
│   ├── .env / .env.example
│   ├── data/
│   │   ├── visdrone_base.yaml
│   │   └── visdrone_augmented.yaml
│   └── runs/                         ← resultados (gitignored)
│       ├── YOLOv12-VisDrone-Base/visdrone_base/
│       └── YOLOv12-VisDrone-Augmented/visdrone_augmented/
└── datasets/                         ← gitignored, poblado por ti
    ├── VisDrone_Base/
    │   ├── images/{train,val}/*.jpg
    │   └── labels/{train,val}/*.txt  (YOLO: class x_center y_center w h, normalizado 0-1)
    └── VisDrone_Augmented/
        ├── images/{train,val}/*.jpg  ← salida de tu pipeline de aumento offline
        └── labels/{train,val}/*.txt
```

- **Clases**: `nc: 1`, `names: ['person']` (índice `0`) en ambos `.yaml`.
- **Experimento 1** (`visdrone_base.yaml`) usa `datasets/VisDrone_Base` (dataset
  convertido a formato YOLO, sin preprocesamiento adicional).
- **Experimento 2** (`visdrone_augmented.yaml`) usa `datasets/VisDrone_Augmented`
  (mismas imágenes de origen, pasadas por tu pipeline de aumento de datos
  *offline*, previo al entrenamiento). Los aumentos *on-the-fly* de YOLO
  (mosaic, mixup, fliplr, hsv, etc.) se aplican igual en ambos casos vía los
  valores por defecto de `ultralytics/cfg/default.yaml` — la única variable
  entre los dos experimentos es el contenido físico de imágenes/etiquetas.

## 5. Hiperparámetros (idénticos en ambos experimentos)

`train_yolo12.py` **no** pasa overrides de `lr0`, `optimizer`, `mosaic`,
`mixup`, `fliplr`, `hsv_*`, `degrees`, etc. — todos se heredan sin modificar
de `ultralytics/cfg/default.yaml`, entre ellos:

| Parámetro | Valor por defecto |
|---|---|
| `optimizer` | `auto` |
| `lr0` / `lrf` | `0.01` / `0.01` |
| `momentum` | `0.937` |
| `weight_decay` | `0.0005` |
| `mosaic` | `1.0` |
| `mixup` | `0.0` |
| `fliplr` | `0.5` |
| `hsv_h/s/v` | `0.015 / 0.7 / 0.4` |
| `close_mosaic` | `10` (últimas 10 épocas sin mosaic) |

Solo se controlan parámetros de **ejecución/hardware** (no de red): `epochs`,
`imgsz`, `batch`, `workers`, `amp`, `device`, `patience` — ver §7.

## 6. Credenciales W&B (seguras, sin hardcodear)

- `.env.example` (versionado en git, sin secretos reales) documenta las
  variables requeridas.
- `.env` (NO versionado, ver `.gitignore`) contiene tu `WANDB_API_KEY` real.
- `train_yolo12.py` carga `.env` con `python-dotenv` y llama a
  `wandb.login(key=...)` antes de entrenar; si `WANDB_API_KEY` falta, el
  script aborta con un mensaje claro en vez de entrenar sin tracking.
- Cada experimento reporta a un **proyecto W&B independiente**:
  `${WANDB_PROJECT}-Base` y `${WANDB_PROJECT}-Augmented`.

## 7. Ejecutar ambos entrenamientos (PowerShell)

```powershell
# Experimento 1: dataset base
python train_yolo12.py `
    --data data\visdrone_base.yaml `
    --name visdrone_base `
    --model yolo12l.pt `
    --epochs 100 `
    --imgsz 1280 `
    --batch 8 `
    --workers 2

# Experimento 2: dataset aumentado (offline)
python train_yolo12.py `
    --data data\visdrone_augmented.yaml `
    --name visdrone_augmented `
    --model yolo12l.pt `
    --epochs 100 `
    --imgsz 1280 `
    --batch 8 `
    --workers 2
```

Resultados guardados en carpetas independientes:
`runs/YOLOv12-VisDrone-Base/visdrone_base/` y
`runs/YOLOv12-VisDrone-Augmented/visdrone_augmented/`.

> **Resolución de imagen (720p)**: las imágenes fuente son 1280×720. En modo
> `train`, ultralytics recibe `imgsz` como un único entero que define el lado
> largo del letterbox cuadrado (aquí `1280`); el lado corto se rellena
> (padding) en vez de recortarse o deformarse, así que no se pierde detalle.
> Con YOLOv12-L, los bloques de atención (`A2C2f` / Area Attention) escalan en
> memoria más que una CNN convencional al subir la resolución, por eso el
> `--batch` por defecto baja de 16 (a 640px) a 8 (a 1280px) para evitar OOM en
> 16 GB de VRAM. Si aun así hay `OOM`, baja `--batch` a `4` o usa `--batch -1`
> (autobatch); si sobra VRAM, puedes subirlo con margen.

Si aparece `BrokenPipeError` / `EOFError` (multiprocessing en Windows), reduce
`--workers` a `0`.

## 8. Métricas registradas en W&B

La integración nativa de ultralytics (`ultralytics/utils/callbacks/wb.py`,
activada vía `settings.update({"wandb": True})`) ya registra automáticamente,
por época:

- `metrics/precision(B)`, `metrics/recall(B)`
- `metrics/mAP50(B)`, `metrics/mAP50-95(B)`
- Pérdidas de entrenamiento: `train/box_loss`, `train/cls_loss`, `train/dfl_loss`
- Curvas Precision-Recall, F1-Confidence, Precision-Confidence,
  Recall-Confidence (una serie por clase; aquí solo `person`)
- Matriz de confusión y artefacto del mejor checkpoint (`best.pt`)

`train_yolo12.py` añade un callback adicional (`on_fit_epoch_end`) que
registra las mismas métricas con nombres explícitos bajo el prefijo
`person/` para lectura directa en el dashboard:

- `person/precision`, `person/recall`, `person/f1_score`
- `person/mAP50`, `person/mAP50-95`
- `person/iou_at_0.5` (= `mAP50`: fracción de detecciones con IoU ≥ 0.5, que
  es la definición operativa de "IoU" a nivel de dataset en detección de
  objetos — no existe un IoU escalar único por época en detección, a
  diferencia de segmentación)
- `person/accuracy` (índice de Jaccard `TP / (TP + FP + FN)` derivado de la
  matriz de confusión 2×2 `person` vs. `background`; es la métrica más
  cercana a "accuracy" en detección de un solo objeto, ya que no existe
  accuracy de clasificación estándar cuando no hay negativos verdaderos
  explícitos por imagen)

## 9. Verificación de GPU (incluida en el script)

Antes de cada entrenamiento, `train_yolo12.py` imprime y valida:

```python
torch.cuda.is_available()
torch.version.cuda
torch.cuda.get_device_name(0)
torch.cuda.get_device_capability(0)
```

Si `torch.cuda.is_available()` es `False`, el script aborta antes de cargar
el dataset o inicializar W&B.
