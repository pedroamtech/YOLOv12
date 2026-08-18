# YOLOv12-Medium sobre VisDrone (Windows) — Documentación de experimentos

Documentación independiente del flujo de entrenamiento local en Windows para
`yolo12m.pt` sobre un dataset VisDrone de clase única (`person`), con tracking
en Weights & Biases. No modifica `requirements.txt` ni ningún archivo del
paquete `ultralytics/` original.

> Se cambió de `yolo12l.pt` (Large) a `yolo12m.pt` (Medium): Large exigía
> demasiado cómputo/VRAM para esta GPU con este dataset (ver el historial de
> ajustes de `imgsz`/`batch` más abajo, motivado por el mismo problema).

## Instrucciones de ejecución

Sigue este orden para reproducir el experimento sin errores:

1. **Preparación del entorno**: crea el entorno conda (§3) e instala las
   dependencias — **no** el `requirements.txt` original del repo, sino
   `requirements-windows.txt` (§4), que es el archivo independiente pensado
   para este flujo en Windows.
2. **Configuración de parámetros**: apunta `data/visdrone_base.yaml` y
   `data/visdrone_augmented.yaml` (§5) a tu dataset real, y copia
   `.env.example` a `.env` (§7) con tu `WANDB_API_KEY`/`WANDB_PROJECT` reales.
   No toques los hiperparámetros de red (§6) — deben quedar idénticos entre
   Experimento 1 y 2.
3. **Ejecución**: corre `train_yolo12.py` con los comandos de PowerShell de
   la §8, uno por experimento (Base y Augmented).
4. **Resolución de problemas**: si algo falla o los resultados difieren de
   lo esperado, revisa la §11 al final de este documento — reúne los
   problemas ya encontrados y resueltos durante estas pruebas (build de
   `stringzilla`, autenticación de W&B, OOM en `TaskAlignedAssigner`,
   entrenamiento lento).

## 1. Hardware y entorno

| Componente        | Valor                              |
|--------------------|-------------------------------------|
| SO                 | Windows 11 Pro                      |
| GPU                | NVIDIA GeForce RTX 5060 Ti — 16 GB VRAM |
| CUDA Toolkit       | 13.3 (driver del sistema)           |
| Framework          | PyTorch + CUDA (wheel, ver §2)      |
| Modelo             | YOLOv12-Medium (`yolo12m.pt`)       |

> **Nota sobre CUDA 13.3 y wheels de PyTorch (confirmado en esta máquina)**: los
> wheels oficiales de PyTorch se distribuyen con etiquetas `cuXXX` (p. ej.
> `cu124`, `cu128`) que empaquetan su propio runtime CUDA; no necesitan
> coincidir exactamente con la versión del CUDA Toolkit del sistema, solo
> requieren un **driver NVIDIA igual o más nuevo** que el mínimo exigido por
> ese runtime. La RTX 5060 Ti es arquitectura **Blackwell (compute capability
> `sm_120`)**. **`cu124` NO sirve**: se probó (`torch==2.6.0+cu124`) y, aunque
> `torch.cuda.is_available()` devuelve `True` (por eso es engañoso — solo
> verifica que hay GPU + driver, no que el build tenga kernels para esa
> arquitectura), PyTorch advierte explícitamente `NVIDIA GeForce RTX 5060 Ti
> with CUDA capability sm_120 is not compatible with the current PyTorch
> installation` (soporta hasta `sm_90`, RTX 40). El fix verificado es
> reinstalar con **`cu128`** (comando exacto en §4).

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

> **Problema conocido en Windows: build de `stringzilla` falla (`Microsoft Visual
> C++ 14.0 or greater is required`)**. `albumentations==2.0.4` depende de
> `albucore==0.0.23`, que a su vez exige `stringzilla>=3.10.4`. Desde su serie
> 2.x, `stringzilla` **dejó de publicar wheels precompilados para Windows** en
> PyPI (los últimos `win_amd64` disponibles son de la serie 1.2.x, por debajo
> del mínimo que pide `albucore`), así que `pip` intenta compilarlo desde el
> código fuente y falla si no tienes el compilador de Microsoft C++ instalado.
> No es un problema de este repo ni de `requirements-windows.txt`: es una
> limitación actual del paquete `stringzilla` en Windows, y **afecta a
> cualquier versión de modelo de ultralytics** (YOLOv12, YOLOv11 `yolo11l.pt`,
> etc.) que use `albumentations` en Windows — no es específico de `yolo12m.pt`.
>
> **Fix (verificado)**: instalar *Build Tools for Visual Studio*
> (https://visualstudio.microsoft.com/visual-cpp-build-tools/) **no es
> suficiente por sí solo** — el instalador base no incluye el compilador de
> C++. Hay que abrir **"Visual Studio Installer"**, elegir **Modificar** sobre
> "Visual Studio Build Tools", y en la pestaña *Workloads* marcar
> explícitamente **"Desktop development with C++"** (que trae MSVC v143 +
> Windows SDK). Sin ese workload marcado, `cl.exe` no existe en el sistema y
> el error persiste aunque el instalador ya se haya "completado". Tras
> instalar el workload, cerrar todas las ventanas de PowerShell abiertas (para
> refrescar el entorno), abrir una nueva, reactivar el entorno conda
> (`conda activate yolov12`) y reintentar `pip install -r
> requirements-windows.txt` — `stringzilla` es código SIMD portable en C/C++ y
> compila sin problemas una vez presente el compilador.

## 3. Creación del entorno virtual (Anaconda)

Todo el trabajo de este flujo (instalación de dependencias, entrenamiento,
tracking) se hace dentro de un entorno conda dedicado, para no interferir con
otras instalaciones de Python/PyTorch en el sistema.

```powershell
# 1) Crear el entorno con Python 3.11 (coincide con la versión objetivo del
#    repo; onnxruntime-gpu, torch y el resto de wheels tienen soporte sólido)
conda create -n yolov12 python=3.11 -y

# 2) Activar el entorno (repetir esto en cada sesión de PowerShell nueva
#    antes de instalar dependencias o lanzar train_yolo12.py)
conda activate yolov12

# 3) Confirmar que el entorno activo es el correcto
python --version
where.exe python
```

> `where.exe python` debe apuntar a una ruta dentro de
> `...\anaconda3\envs\yolov12\python.exe` (o `...\miniconda3\envs\...`).
> Si apunta al Python global del sistema, el entorno no está activado.

Para desactivar el entorno al terminar la sesión: `conda deactivate`.
Para eliminarlo por completo (p. ej. para reinstalar desde cero):
`conda env remove -n yolov12`.

## 4. Instalación manual en Windows (ningún script automático)

Ejecutar en PowerShell, con el entorno `yolov12` del §3 ya activado:

```powershell
# 1) Actualizar pip
python -m pip install --upgrade pip

# 2) Instalar PyTorch con soporte CUDA — cu128, CONFIRMADO para RTX 5060 Ti / Blackwell
#    (cu124 se probó y NO funciona: PyTorch reporta sm_120 como no soportado)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 2b) SOLO SI cu128 estable también falla (variante nightly, no debería hacer falta):
# pip uninstall -y torch torchvision torchaudio
# pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# 3) Instalar el resto de dependencias (archivo independiente, sin tocar requirements.txt)
pip install -r requirements-windows.txt

# 4) Instalar este repo (YOLOv12) en modo editable
pip install -e .
```

> Evita fijar `torch==2.4.0` en Windows: es una versión con errores conocidos
> en CPU/Windows (ver `pyproject.toml:73`). El comando del paso 2 (índice
> `cu128`) ya resuelve a una versión reciente (`2.7.x`+) que evita ese problema
> y sí soporta Blackwell — no hace falta fijar la versión manualmente.

## 5. Estructura de directorios esperada

Ultralytics resuelve el campo `path` de cada `.yaml` de dataset como
`(datasets_dir / path).resolve()`, donde `datasets_dir` es la carpeta
configurada en `ultralytics/settings.json` (por defecto, la carpeta hermana
`datasets/` junto al repo clonado). Estructura esperada:

```
GitHub/
├── yolov12/                          ← este repo clonado
│   ├── requirements.txt              (original, sin tocar)
│   ├── requirements-windows.txt      (nuevo)
│   ├── train_yolo12.py               (nuevo)
│   ├── README_EXPERIMENTS.md         (este archivo)
│   ├── .env / .env.example
│   ├── data/
│   │   ├── visdrone_base.yaml
│   │   └── visdrone_augmented.yaml
│   └── runs/                         ← resultados (gitignored)
│       └── YOLOv12/                  ← un solo proyecto W&B; Base/Augmented se distinguen por --name
│           ├── visdrone_base/
│           └── visdrone_augmented/
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

## 6. Hiperparámetros (idénticos en ambos experimentos)

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
`imgsz`, `batch`, `workers`, `amp`, `device`, `patience` — ver §8.

## 7. Credenciales W&B (seguras, sin hardcodear)

- `.env.example` (versionado en git, sin secretos reales) documenta las
  variables requeridas.
- `.env` (NO versionado, ver `.gitignore`) contiene tu `WANDB_API_KEY` real.
- `train_yolo12.py` carga `.env` con `python-dotenv`; si `WANDB_API_KEY` falta,
  el script aborta con un mensaje claro en vez de entrenar sin tracking.
- Ambos experimentos reportan al **mismo proyecto W&B** (`${WANDB_PROJECT}`,
  p. ej. `YOLOv12`); Base vs. Augmented se distingue por el **nombre de la
  corrida** (`--name visdrone_base` / `--name visdrone_augmented`), no por el
  proyecto.

> **Por qué no se usa `wandb.login(key=...)` (error corregido)**: esa función
> escribe la key en `~/.netrc` y valida que tenga exactamente 40 caracteres —
> el formato clásico de API key personal (`https://wandb.ai/authorize`). Con
> keys más nuevas con prefijo (p. ej. `wandb_v1_...`, típicas de cuentas de
> servicio/organización) falla con
> `ValueError: API key must be 40 characters long, yours was 86` aunque la key
> sea completamente válida. El script en su lugar exporta
> `os.environ["WANDB_API_KEY"]` directamente, que `wandb.init()` toma sin
> pasar por esa validación de longitud, y autentica contra el backend real.
>
> **Por qué se llama a `wandb.init()` explícitamente en el script (error
> corregido)**: el callback nativo de ultralytics (`wb.py`) deriva el nombre
> de proyecto de W&B a partir del `project=` que se le pasa a `model.train()`
> — que en este script es una ruta local de carpeta
> (`runs\YOLOv12\...`). Ese callback solo limpia el carácter `/`, no `\` ni
> `:`, así que en Windows termina pasándole a W&B un nombre de proyecto como
> `C:\Users\...\runs\YOLOv12`, y W&B lo rechaza:
> `UsageError: Invalid project name '...': cannot contain characters
> '/,\,#,?,%,:'`. El fix es inicializar W&B nosotros mismos, antes de
> `model.train()`, con `project=${WANDB_PROJECT}` (el proyecto único y limpio,
> sin caracteres de ruta) y `name=` la corrida — el callback nativo detecta
> que ya hay un run activo (`wb.run`) y solo loguea métricas en él, sin
> volver a llamar a `wb.init()`.

## 8. Ejecutar ambos entrenamientos (PowerShell)

Con el entorno `yolov12` (§3) activado:

```powershell
# Experimento 1: dataset base
python train_yolo12.py `
    --data data\visdrone_base.yaml `
    --name visdrone_base `
    --model yolo12m.pt `
    --epochs 250 `
    --imgsz 640 `
    --batch 16 `
    --workers 8

# Experimento 2: dataset aumentado (offline)
python train_yolo12.py `
    --data data\visdrone_augmented.yaml `
    --name visdrone_augmented `
    --model yolo12m.pt `
    --epochs 250 `
    --imgsz 640 `
    --batch 16 `
    --workers 8
```

Resultados locales en subcarpetas independientes dentro del mismo proyecto:
`runs/YOLOv12/visdrone_base/` y `runs/YOLOv12/visdrone_augmented/`. En W&B,
ambas corridas caen en el mismo proyecto (`YOLOv12`), distinguidas por nombre
de corrida.

> **Resolución de imagen (bajada al estándar YOLO, 640)**: las imágenes fuente
> son 1280×720; `imgsz=640` es el default de `ultralytics/cfg/default.yaml` y
> el tamaño con el que estos modelos fueron ajustados originalmente (COCO). Se
> probó primero a `1280` (nativo) y luego `960` (punto medio) buscando
> conservar más detalle para las personas pequeñas/lejanas de VisDrone, pero
> ambos tamaños son bastante más pesados en memoria y velocidad con YOLOv12
> (los bloques de atención `A2C2f` / Area Attention escalan peor con
> resolución que una CNN convencional) — `640` prioriza velocidad y
> estabilidad de entrenamiento sobre ese detalle extra. En modo `train`,
> ultralytics recibe `imgsz` como un único entero que define el lado largo
> del letterbox cuadrado; el lado corto se rellena (padding) en vez de
> recortarse o deformarse. **Confirmado en este dataset a `imgsz=1280` con
> `yolo12l.pt` (Large)**: `batch=8` produce `CUDA OutOfMemoryError` en
> `TaskAlignedAssigner` (VisDrone tiene muchísimas cajas por imagen, lo que
> infla el tensor de costo de asignación) — uno de los motivos por los que se
> pasó a `yolo12m.pt` (Medium), bastante más liviano. A `640px` hay mucho más
> margen de VRAM (~6.25× menos píxeles que a 1280px) y el modelo es más
> chico, así que `--batch` sube a `16` — no verificado todavía en este
> dataset con `yolo12m.pt`; si da `OOM`, baja o usa `--batch -1` (autobatch,
> deja que ultralytics mida la VRAM libre real) — solo usa el **mismo** valor
> en ambos experimentos.

> **"El entrenamiento es muy lento / no avanza" (causa raíz confirmada)**: si
> ves `WARNING: CUDA OutOfMemoryError in TaskAlignedAssigner, using CPU` justo
> al arrancar la época 1, **esa es la causa** — no un cuelgue. Ultralytics
> atrapa el `OutOfMemoryError` en ese paso puntual y hace fallback silencioso
> a CPU (mueve los tensores GPU→CPU, calcula ahí, los regresa a GPU), **en
> cada iteración**, lo que hace que la GPU se vea casi al límite de uso pero
> el entrenamiento avance extremadamente lento. Esto fue justo lo que pasó
> con `batch=8` a `imgsz=1280` en este dataset — de ahí que se bajara primero
> el `imgsz` (960, luego 640) y el `batch` se reajustara en cada paso (ver
> nota anterior). Si ya bajaste el batch y sigue lento sin ese warning
> específico, entran en juego motivos normales de rendimiento: los bloques de
> atención de YOLOv12 escalan peor con resolución que una CNN normal, y sin
> FlashAttention (mensaje `"FlashAttention is not available on this device.
> Using scaled_dot_product_attention instead."`, normal en Windows) el
> fallback `scaled_dot_product_attention` es más lento — en una GPU Blackwell
> tan reciente como la RTX 5060 Ti, los kernels siguen madurando. Revisa el
> `s/it` / `ETA` de la barra de progreso: si el ETA es de horas por época, es
> lento, no un cuelgue. `--workers` por defecto es `8` — en una CPU con varios
> núcleos, un valor bajo deja el preprocesamiento (mosaic + albumentations a
> 1280px) como cuello de botella. Ajusta `--workers` según los núcleos lógicos
> de tu CPU (el script avisa si lo pasas por encima); si aparece
> `BrokenPipeError` / `EOFError` (multiprocessing en Windows), baja
> `--workers` a `0`.

## 9. Métricas registradas en W&B

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

## 10. Verificación de GPU (incluida en el script)

Antes de cada entrenamiento, `train_yolo12.py` imprime y valida:

```python
torch.cuda.is_available()
torch.version.cuda
torch.cuda.get_device_name(0)
torch.cuda.get_device_capability(0)
```

Si `torch.cuda.is_available()` es `False`, el script aborta antes de cargar
el dataset o inicializar W&B.

## 11. Resolución de problemas

Problemas reales encontrados y resueltos durante estas pruebas, en el orden
en que suelen aparecer. Cada fila tiene la explicación completa en la
sección indicada.

| Síntoma | Causa | Fix | Detalle |
|---|---|---|---|
| `pip install -r requirements-windows.txt` falla con `Building wheel for stringzilla ... Microsoft Visual C++ 14.0 or greater is required` | `albumentations` arrastra `albucore`→`stringzilla>=3.10.4`, que no publica wheel para Windows desde su serie 2.x | Instalar *Build Tools for Visual Studio* y marcar explícitamente el workload **"Desktop development with C++"** (el instalador base solo, sin ese workload, no basta) | §2 |
| `ValueError: API key must be 40 characters long, yours was 86` al iniciar el entrenamiento | `wandb.login(key=...)` valida el formato clásico de key personal (40 caracteres); las keys con prefijo (`wandb_v1_...`, de cuentas de servicio/organización) no lo cumplen aunque sean válidas | El script ya no llama a `wandb.login()`; exporta `WANDB_API_KEY` como variable de entorno y deja que `wandb.init()` autentique contra el backend real | §7 |
| `wandb.errors.UsageError: Invalid project name '...': cannot contain characters '/,\,#,?,%,:'` | El callback nativo de ultralytics derivaba el nombre de proyecto de W&B a partir de una ruta local de Windows (con `\` y `:`) | El script llama a `wandb.init(project=..., name=...)` con el nombre de proyecto limpio antes de `model.train()` | §7 |
| GPU casi al 100% de uso pero el entrenamiento no avanza (época 1 pegada) | `WARNING: CUDA OutOfMemoryError in TaskAlignedAssigner, using CPU` — `batch`/`imgsz` demasiado altos para la VRAM disponible con este dataset (VisDrone tiene muchísimas cajas por imagen) | Bajar `--batch` y/o `--imgsz`, o usar `--batch -1` (autobatch); también se cambió el modelo de `yolo12l.pt` a `yolo12m.pt` | §1 (nota de imgsz/batch), historial de commits |
| Entrenamiento lento pero **sin** ese warning de OOM | Normal a mayor resolución con YOLOv12: los bloques de atención (`A2C2f`) escalan peor que una CNN, y sin FlashAttention (`"Using scaled_dot_product_attention instead"`, esperado en Windows) el fallback es más lento — más notorio en una GPU Blackwell reciente con kernels aún inmaduros | Revisar el `s/it`/`ETA` de la barra de progreso antes de asumir un cuelgue; considerar bajar `imgsz` o subir `--workers` si el cuello de botella es el preprocesamiento en CPU | §1 |
| `torch.cuda.is_available()` da `True` pero el entrenamiento falla o cae a CPU sin avisar | El wheel de PyTorch instalado (`cu124`) no incluye kernels para Blackwell (`sm_120`, RTX 50-series) | Reinstalar con `--index-url https://download.pytorch.org/whl/cu128` | §1, §4 |

