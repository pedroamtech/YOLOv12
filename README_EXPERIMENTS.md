# YOLOv12 Nano/Small sobre VisDrone (Windows) — Documentación de experimentos

Este documento describe el flujo de entrenamiento local en Windows armado
para `yolo12n.pt`/`yolo12s.pt` sobre un dataset VisDrone de clase única
(`person`), con tracking en Weights & Biases. No modifica `requirements.txt`
ni ningún archivo del paquete `ultralytics/` original.

> **Alcance actual: solo Nano y Small.** Se probó primero con `yolo12l.pt`
> (Large) — exigía demasiado cómputo/VRAM para esta GPU con este dataset
> (ver el historial de ajustes de `imgsz`/`batch` más abajo) — después con
> `yolo12m.pt` (Medium), y finalmente se redujo el alcance del proyecto a
> Nano y Small únicamente. `--model yolo12m.pt` (o `yolo12l.pt`) sigue
> siendo técnicamente válido para retomarlos: el script no cambió, solo el
> plan de experimentos documentado acá.

## Instrucciones de ejecución

Sigue este orden para reproducir el experimento sin errores:

1. **Preparación del entorno**: crea el entorno conda (sección 3) e instala
   las dependencias — **no** el `requirements.txt` original del repo, sino
   `requirements-windows.txt` (sección 4), el archivo independiente armado
   para este flujo en Windows.
2. **Configuración de parámetros**: apunta `data/visdrone_base.yaml` y
   `data/visdrone_augmented.yaml` (sección 5) a tu dataset real, y copia
   `.env.example` a `.env` (sección 8) con tu `WANDB_API_KEY`/`WANDB_PROJECT`
   reales. No toques los hiperparámetros de red (sección 6) — quedan
   idénticos en las cuatro corridas.
3. **Ejecución**: corre `train_yolo12.py` con los comandos de PowerShell de
   la sección 9 — cuatro corridas: Nano y Small, cada uno con Base y
   Augmented.
4. **Resolución de problemas**: si algo falla o los resultados difieren de
   lo esperado, revisa la sección 12 al final de este documento — reúne los
   problemas encontrados y resueltos durante estas pruebas (build de
   `stringzilla`, longitud de la API key de W&B, nombre de proyecto de W&B
   con caracteres inválidos, OOM en `TaskAlignedAssigner`, entrenamiento
   lento, `cu124` incompatible con Blackwell).

## 1. Hardware y entorno

| Componente        | Valor                              |
|--------------------|-------------------------------------|
| SO                 | Windows 11 Pro                      |
| GPU                | NVIDIA GeForce RTX 5060 Ti — 16 GB VRAM |
| CUDA Toolkit       | 13.3 (driver del sistema)           |
| Framework          | PyTorch + CUDA (wheel, ver sección 2)      |
| Modelo             | YOLOv12 Nano (`yolo12n.pt`) y Small (`yolo12s.pt`) |

> **Nota sobre CUDA 13.3 y wheels de PyTorch (confirmado en esta máquina)**:
> los wheels oficiales de PyTorch se distribuyen con etiquetas `cuXXX` (p.
> ej. `cu124`, `cu128`) que empaquetan su propio runtime CUDA; no necesitan
> coincidir exactamente con la versión del CUDA Toolkit del sistema, solo
> requieren un **driver NVIDIA igual o más nuevo** que el mínimo exigido por
> ese runtime. La RTX 5060 Ti es arquitectura **Blackwell (compute
> capability `sm_120`)**. **`cu124` NO sirve**: se probó
> (`torch==2.6.0+cu124`) y, aunque `torch.cuda.is_available()` devuelve
> `True` (por eso es engañoso — solo verifica que hay GPU + driver, no que
> el build tenga kernels para esa arquitectura), PyTorch advierte
> explícitamente `NVIDIA GeForce RTX 5060 Ti with CUDA capability sm_120 is
> not compatible with the current PyTorch installation` (lista textual de
> capacidades soportadas por ese build: `sm_50 sm_60 sm_61 sm_70 sm_75 sm_80
> sm_86 sm_90` — `sm_86` es RTX 30, la última serie de consumo listada;
> `sm_90` es Hopper/H100, de datacenter, no RTX 40 (`sm_89`), que ni
> siquiera aparece en la lista). El fix confirmado es reinstalar con
> **`cu128`** (comando
> exacto en sección 4).

## 2. Diferencias: `requirements.txt` (original) vs `requirements-windows.txt` (nuevo)

| Paquete | `requirements.txt` (original) | `requirements-windows.txt` (Windows) | Motivo |
|---|---|---|---|
| `torch` / `torchvision` | Pineado (`torch==2.2.2`) | **Excluido del archivo** — se instala aparte | Necesita un build reciente con soporte `sm_120` (Blackwell) y el índice CUDA correcto; pinear una versión antigua rompe la RTX 5060 Ti |
| `flash_attn` | Wheel Linux `cp311-linux_x86_64` | **Eliminado** | No existe wheel oficial para Windows; no hace falta, YOLOv12 usa `torch.nn.functional.scaled_dot_product_attention` como fallback |
| `onnxruntime` (CPU) | Incluido junto a `onnxruntime-gpu` | **Eliminado** | Evita conflicto de paquete duplicado (CPU vs GPU) en Windows |
| `onnx` | `1.14.0` | `1.16.1` | Compatibilidad con export en Windows/Python recientes |
| `python-dotenv` | No estaba | **Añadido** | Carga segura de `WANDB_API_KEY` desde `.env` |
| `wandb` | No estaba | **Añadido** | Tracking de métricas |
| Resto (`timm`, `albumentations`, `pycocotools`, `opencv-python`, etc.) | igual | igual | Sin cambios, multiplataforma |

> **Problema conocido en Windows: build de `stringzilla` falla (`Microsoft
> Visual C++ 14.0 or greater is required`)**. `albumentations==2.0.4`
> depende de `albucore==0.0.23`, que a su vez exige `stringzilla>=3.10.4`.
> Desde su serie 2.x, `stringzilla` **dejó de publicar wheels precompilados
> para Windows** en PyPI (los últimos `win_amd64` disponibles son de la
> serie 1.2.x, por debajo del mínimo que pide `albucore`), así que `pip`
> intenta compilarlo desde el código fuente y falla sin el compilador de
> Microsoft C++ instalado. No es un problema de este repo ni de
> `requirements-windows.txt`: es una limitación actual del paquete
> `stringzilla` en Windows, y **afecta a cualquier versión de modelo de
> ultralytics** (YOLOv12, YOLOv11 `yolo11l.pt`, etc.) que use
> `albumentations` en Windows — no es específico de ningún tamaño de
> YOLOv12 en particular (`yolo12n.pt`, `yolo12s.pt`, etc.).
>
> **Fix (verificado)**: instalar *Build Tools for Visual Studio*
> (https://visualstudio.microsoft.com/visual-cpp-build-tools/) **no basta
> por sí solo** — el instalador base no incluye el compilador de C++. Abre
> **"Visual Studio Installer"**, elige **Modificar** sobre "Visual Studio
> Build Tools", y en la pestaña *Workloads* marca explícitamente
> **"Desktop development with C++"** (trae MSVC v143 + Windows SDK). Sin ese
> workload marcado, `cl.exe` no existe en el sistema y el error persiste
> aunque el instalador ya se haya "completado". Después de instalar el
> workload, cierra todas las ventanas de PowerShell abiertas (para refrescar
> el entorno), abre una nueva, activa el entorno conda (`conda activate
> yolov12`) y reintenta `pip install -r requirements-windows.txt` —
> `stringzilla` es código SIMD portable en C/C++ y compila sin problemas una
> vez presente el compilador.

## 3. Creación del entorno virtual (Anaconda)

Todo este flujo (instalación de dependencias, entrenamiento, tracking) se
arma dentro de un entorno conda dedicado, para no interferir con otras
instalaciones de Python/PyTorch en el sistema.

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

> `where.exe python` apunta a una ruta dentro de
> `...\anaconda3\envs\yolov12\python.exe` (o `...\miniconda3\envs\...`). Si
> apunta al Python global del sistema, el entorno no está activado.

Para desactivar el entorno al terminar la sesión: `conda deactivate`. Para
eliminarlo por completo (reinstalación desde cero): `conda env remove -n
yolov12`.

## 4. Instalación manual en Windows (ningún script automático)

Con el entorno `yolov12` de la sección 3 activado, ejecuta en PowerShell:

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

> Se evita fijar `torch==2.4.0` en Windows: es una versión con errores
> conocidos en CPU/Windows (ver `pyproject.toml:73`). El comando del paso 2
> (índice `cu128`) ya resuelve a una versión reciente (`2.7.x`+) que evita
> ese problema y sí soporta Blackwell — no hace falta fijar la versión
> manualmente.

## 5. Estructura de directorios esperada

`data/visdrone_base.yaml` (dataset base) usa una ruta **absoluta** en
`path:`, así que ultralytics la usa tal cual, sin pasar por `datasets_dir`.
Apunta al dataset real, confirmado en disco:

```
C:\Users\pedroam\Documents\Data-Augmentation\Datasets-Clean\VisDrone\
├── train\
│   ├── images\        (6890 imágenes)
│   └── labels\         (6890 .txt, YOLO: class x_center y_center w h, normalizado 0-1)
└── val\
    ├── images\        (1723 imágenes)
    └── labels\         (1723 .txt)
```

> **Partición 80/20**: el dataset base quedó reparticionado en 6890 imágenes
> de entrenamiento y 1723 de validación sobre un total de 8613 — exactamente
> 80.00%/20.00%.

`data/visdrone_augmented.yaml` (dataset aumentado) **todavía usa una ruta
relativa placeholder** (`../datasets/VisDrone_Augmented`, resuelta por
ultralytics como `(datasets_dir / path).resolve()`, donde `datasets_dir` es
la carpeta configurada en `ultralytics/settings.json` — por defecto, la
carpeta hermana `datasets/` junto al repo clonado). Actualízala del mismo
modo que `visdrone_base.yaml` (path absoluto) en cuanto esté lista la copia
con el pipeline de aumento offline aplicado:

```
GitHub/
├── yolov12/                          ← este repo clonado
│   ├── requirements.txt              (original, sin tocar)
│   ├── requirements-windows.txt      (nuevo)
│   ├── train_yolo12.py               (nuevo)
│   ├── README_EXPERIMENTS.md         (este archivo)
│   ├── .env / .env.example
│   ├── data/
│   │   ├── visdrone_base.yaml        (path absoluto, ver arriba)
│   │   └── visdrone_augmented.yaml   (pendiente de ruta real)
│   └── runs/                         ← resultados (gitignored)
│       └── YOLOv12/                  ← un solo proyecto W&B; modelo/dataset se distinguen por --name
│           ├── nano_base/
│           ├── nano_augmented/
│           ├── small_base/
│           └── small_augmented/
└── datasets/                         ← gitignored; solo necesario para el dataset aumentado
    └── VisDrone_Augmented/
        ├── images/{train,val}/*.jpg  ← salida del pipeline de aumento offline
        └── labels/{train,val}/*.txt
```

- **Clases**: `nc: 1`, `names: ['person']` (índice `0`) en ambos `.yaml`.
  Verificado en el dataset real: las etiquetas solo usan el índice `0`.
- **Dataset base** (`visdrone_base.yaml`) apunta al dataset de arriba, sin
  preprocesamiento adicional.
- **Dataset aumentado** (`visdrone_augmented.yaml`) apunta a una copia de
  las mismas imágenes, pasadas por un pipeline de aumento de datos
  *offline*, previo al entrenamiento. Los aumentos *on-the-fly* de YOLO
  (mosaic, mixup, fliplr, hsv, etc. — sección 6) se aplican igual en ambos
  casos, con los mismos valores por defecto de `ultralytics/cfg/default.yaml`
  en las cuatro corridas — la única variable entre dataset base y aumentado
  es el contenido físico de imágenes/etiquetas.

## 6. Hiperparámetros (idénticos en las cuatro corridas)

`train_yolo12.py` **no** pasa overrides de ningún hiperparámetro de red —
todos se heredan sin modificar de `ultralytics/cfg/default.yaml`. Agrupados
por categoría:

### 6.1 Optimización / entrenamiento

| Parámetro | Valor por defecto |
|---|---|
| `optimizer` | `auto` |
| `lr0` / `lrf` | `0.01` / `0.01` |
| `momentum` | `0.937` |
| `weight_decay` | `0.0005` |
| `warmup_epochs` | `3.0` |
| `warmup_momentum` | `0.8` |
| `warmup_bias_lr` | `0.0` |
| `cos_lr` | `False` (decay lineal, no coseno) |

> Estos son los valores tal cual están en `ultralytics/cfg/default.yaml` —
> con `optimizer: auto`, varios se recalculan o se sobreescriben en tiempo
> de ejecución (`optimizer`, `momentum` efectivo). El detalle completo, con
> líneas de código exactas, está en la sección 7.

### 6.2 Función de pérdida

| Parámetro | Valor por defecto |
|---|---|
| `box` | `7.5` (peso de la pérdida de caja) |
| `cls` | `0.5` (peso de la pérdida de clase) |
| `dfl` | `1.5` (peso de distribution focal loss) |

### 6.3 Aumento de datos clásico (on-the-fly, activo en detección)

| Parámetro | Valor por defecto |
|---|---|
| `hsv_h` / `hsv_s` / `hsv_v` | `0.015` / `0.7` / `0.4` |
| `degrees` | `0.0` (rotación) |
| `translate` | `0.1` |
| `scale` | `0.5` |
| `shear` | `0.0` |
| `perspective` | `0.0` |
| `flipud` | `0.0` (flip vertical) |
| `fliplr` | `0.5` (flip horizontal) |
| `bgr` | `0.0` |
| `mosaic` | `1.0` |
| `mixup` | `0.0` |
| `copy_paste` | `0.1` (configurado, pero **inerte** en este proyecto — ver nota abajo) |
| `copy_paste_mode` | `flip` |
| `close_mosaic` | `10` (últimas 10 épocas sin mosaic) |

> **`copy_paste` no tiene efecto real en estos experimentos**: en
> `ultralytics/data/augment.py:1676`, la clase `CopyPaste` hace
> `if len(labels["instances"].segments) == 0 or self.p == 0: return labels`
> — es un no-op si las labels no tienen `segments` (máscaras de
> segmentación). El dataset VisDrone de este proyecto son solo bounding
> boxes (formato YOLO `class x_center y_center w h`, sin polígonos), así
> que `segments` siempre tiene longitud 0 y `copy_paste` nunca se ejecuta,
> sin importar el valor `0.1` configurado. A diferencia de `mixup: 0.0` o
> `shear: 0.0` (apagados por valor), acá el valor sugiere que está
> "encendido" pero es inerte por incompatibilidad de formato de datos.

### 6.4 Aumento adicional vía Albumentations (fijo en código, no en `default.yaml`)

Además de lo anterior, ultralytics aplica siempre esta transformación
`albumentations` — visible en el log de cada corrida
(`albumentations: Blur(p=0.01, ...), MedianBlur(p=0.01, ...), ToGray(p=0.01,
...), CLAHE(p=0.01, ...)`) — con probabilidades **hardcodeadas** en
`ultralytics/data/augment.py:1847-1853`, no configurables vía
`default.yaml` ni CLI. La composición completa tiene 7 transforms, 4 con
probabilidad activa y 3 en `p=0.0` (presentes en el código pero sin efecto
observable):

| Transform | Probabilidad |
|---|---|
| `Blur` | `0.01` |
| `MedianBlur` | `0.01` |
| `ToGray` | `0.01` |
| `CLAHE` | `0.01` |
| `RandomBrightnessContrast` | `0.0` (sin efecto) |
| `RandomGamma` | `0.0` (sin efecto) |
| `ImageCompression` | `0.0` (sin efecto) |

Es idéntica en las cuatro corridas por ser parte fija del código, no de los
hiperparámetros.

> **No aplican a estos experimentos**: `auto_augment`, `erasing` y
> `crop_fraction` también existen en `default.yaml` y aparecen en el log de
> `args` de cada corrida, pero según su propia documentación en el archivo
> son específicos de **clasificación** (`ultralytics/data/dataset.py`) — no
> afectan el pipeline de aumento de detección que usan estos experimentos.

Solo se controlan parámetros de **ejecución/hardware** (no de red):
`epochs`, `imgsz`, `batch`, `workers`, `amp`, `device`, `patience` — ver
sección 9.

## 7. Metodología de entrenamiento (transfer learning / fine-tuning)

- **Transfer learning + fine-tuning desde pesos preentrenados en COCO, no
  entrenamiento desde cero.** `train_yolo12.py` siempre instancia
  `YOLO(args.model)` con un checkpoint `.pt` (`yolo12n.pt`/`yolo12s.pt`),
  nunca con un `.yaml` de arquitectura sin entrenar. En
  `ultralytics/engine/trainer.py:578-580` (`BaseTrainer.setup_model`):
  cuando `self.model` termina en `.pt`, llama a `attempt_load_one_weight`
  para cargar esos pesos como punto de partida — no hay inicialización
  aleatoria. Como VisDrone tiene una sola clase (`person`) contra las 80 de
  COCO, la cabeza de clasificación no calza 1:1 con el checkpoint — el log
  de cada corrida muestra estas dos líneas, generadas por
  `ultralytics/nn/tasks.py:315-317` y `ultralytics/nn/tasks.py:265-278`:

  ```
  Overriding model.yaml nc=80 with nc=1
  Transferred 1031/1341 items from pretrained weights
  ```

  (el segundo número es de una corrida real con `yolo12l.pt` — varía según
  Nano/Small; lo importante es que **no** dice 1341/1341: `nn/tasks.py:275`
  hace `intersect_dicts(csd, self.state_dict())` antes de cargar los pesos,
  así que cualquier tensor cuya forma no calce — los canales de
  clasificación del head, dimensionados para 80 clases — se descarta, y el
  resto del backbone y el neck sí se transfieren completos).

- **Ninguna capa congelada por configuración — fine-tuning general desde la
  época 1, con una excepción fija por diseño.** `train_yolo12.py` no pasa
  `freeze=` a `model.train()`, así que se usa el default de ultralytics:
  `freeze=None` → `freeze_list=[]` en `ultralytics/engine/trainer.py:238-244`
  — no hay una fase inicial con backbone congelado ni un "unfreeze"
  progresivo. Pero **un módulo queda siempre congelado sin importar
  `freeze=`**: `ultralytics/engine/trainer.py:246-251` fija
  `always_freeze_names = [".dfl"]` — la proyección de distribution focal
  loss que convierte la distribución de probabilidad de cada borde de caja
  en una coordenada, hardcodeada como no entrenable por diseño de la
  arquitectura (visible en el log como `Freezing layer
  'model.21.dfl.conv.weight'`). El resto de la red ajusta el 100% de sus
  parámetros.

- **Hiperparámetros clave del fine-tuning** (ya listados en la sección 6.1;
  acá el detalle de dónde el valor *real* en tiempo de ejecución difiere
  del que aparece en `default.yaml`, por el propio comportamiento de
  `optimizer: auto`):

  | Parámetro | Valor en `default.yaml` | Qué pasa realmente en tiempo de ejecución |
  |---|---|---|
  | `optimizer` | `auto` | `build_optimizer` (`ultralytics/engine/trainer.py:759-788`): con `nc=1`, `epochs=250`, `batch=16` (`nbs=64`), `iterations = ceil(6890/64) * 250 = 27.000`, muy por encima del umbral de `10.000` que decide entre SGD y AdamW — resuelve a **`SGD(lr=0.01, momentum=0.9)`** |
  | `lr0` | `0.01` | Coincide con el `lr` real solo porque la rama SGD de `optimizer: auto` también hardcodea `lr=0.01` — no es un traspaso directo del valor de `default.yaml` |
  | `lrf` | `0.01` | Fracción final: la LR decae hasta `lr0 × lrf = 0.0001` al terminar las 250 épocas |
  | `momentum` | `0.937` | **Se ignora al construir el optimizador**: la rama SGD de `optimizer: auto` hardcodea `momentum=0.9` en una variable local (`trainer.py:787`), sin reescribir `self.args.momentum` — confirmado en el log: `ignoring 'lr0=0.01' and 'momentum=0.937' [...] determining best [...] automatically`. El `SGD(lr=0.01, momentum=0.9)` que se imprime al arrancar es solo el valor de construcción — ver la fila `warmup_momentum` para el valor que termina quedando vigente |
  | `cos_lr` | `False` | Scheduler **lineal** (`_setup_scheduler`, `ultralytics/engine/trainer.py:209-215`), no coseno: `lr(epoch) = lr0 × (max(1 − epoch/epochs, 0) × (1 − lrf) + lrf)` |
  | `warmup_epochs` | `3.0` | Sin cambios — las primeras 3 épocas interpolan LR y momentum en vez de arrancar de golpe |
  | `warmup_momentum` | `0.8` | **El momentum termina en `0.937`, no en `0.9`.** `trainer.py:376` hace `x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])`, escribiendo directamente sobre `self.optimizer.param_groups` — y `self.args.momentum` sigue en `0.937` (nunca se reescribió, ver fila `momentum`). Como `self.args.momentum` es el valor objetivo de esa interpolación, el warmup lleva el momentum real del optimizador de `0.8` a `0.937` a lo largo de las 3 épocas, no a `0.9`. Ningún otro punto del código vuelve a tocar `param_groups[...]["momentum"]`, así que `0.937` queda fijo el resto de las 250 épocas — el `momentum=0.9` del log solo es cierto en el instante de construcción, antes de que arranque el warmup |
  | `warmup_bias_lr` | `0.0` | `build_optimizer` fija explícitamente `self.args.warmup_bias_lr = 0.0` en la rama `auto` (`ultralytics/engine/trainer.py:788`) — coincide con el default de `default.yaml` en este caso, así que no hay cambio observable |

  Además, `accumulate = max(round(nbs / batch), 1) = max(round(64/16), 1) = 4`
  (`ultralytics/engine/trainer.py:301`): acumula gradiente de 4 batches
  antes de cada paso de optimización, para aproximar un batch nominal de 64
  aunque `--batch` sea 16. `weight_decay` se escala por
  `batch × accumulate / nbs = 16×4/64 = 1` (`trainer.py:302`), así que queda
  igual a `0.0005` con esta combinación de `batch`/`nbs`.

  Esta combinación (fine-tuning general con `.dfl` siempre congelado,
  warmup de 3 épocas, decaimiento lineal, optimizador auto-resuelto a SGD)
  es la misma en las cuatro corridas — Nano y Small parten de sus
  respectivos checkpoints de COCO, no de una arquitectura sin entrenar.

## 8. Credenciales W&B (seguras, sin hardcodear)

- `.env.example` (versionado en git, sin secretos reales) documenta las
  variables requeridas.
- `.env` (NO versionado, ver `.gitignore`) contiene la `WANDB_API_KEY` real.
- `train_yolo12.py` carga `.env` con `python-dotenv`; si `WANDB_API_KEY`
  falta, el script aborta con un mensaje claro en vez de entrenar sin
  tracking.
- Las cuatro corridas reportan al **mismo proyecto W&B** (`${WANDB_PROJECT}`,
  p. ej. `YOLOv12`); modelo (Nano/Small) y dataset (Base/Augmented) se
  distinguen por el **nombre de la corrida** (`--name nano_base`,
  `nano_augmented`, `small_base`, `small_augmented`), no por
  el proyecto.

> **Por qué el script no usa `wandb.login(key=...)` (error corregido)**: esa
> función escribe la key en `~/.netrc` y valida que tenga exactamente 40
> caracteres — el formato clásico de API key personal
> (`https://wandb.ai/authorize`). Con keys más nuevas con prefijo (p. ej.
> `wandb_v1_...`, típicas de cuentas de servicio/organización) falla con
> `ValueError: API key must be 40 characters long, yours was 86` aunque la
> key sea completamente válida. El script exporta `os.environ["WANDB_API_KEY"]`
> directamente, que `wandb.init()` toma sin pasar por esa validación de
> longitud, y autentica contra el backend real.
>
> **Por qué el script llama a `wandb.init()` explícitamente (error
> corregido)**: el callback nativo de ultralytics (`wb.py`) deriva el nombre
> de proyecto de W&B a partir del `project=` que se le pasa a
> `model.train()` — que en este script es una ruta local de carpeta
> (`runs\YOLOv12\...`). Ese callback solo limpia el carácter `/`, no `\` ni
> `:`, así que en Windows termina pasándole a W&B un nombre de proyecto como
> `C:\Users\...\runs\YOLOv12`, y W&B lo rechaza:
> `UsageError: Invalid project name '...': cannot contain characters
> '/,\,#,?,%,:'`. El fix es inicializar W&B antes de `model.train()`, con
> `project=${WANDB_PROJECT}` (el proyecto único y limpio, sin caracteres de
> ruta) y `name=` la corrida — el callback nativo detecta que ya hay un run
> activo (`wb.run`) y solo loguea métricas en él, sin volver a llamar a
> `wb.init()`.

## 9. Ejecutar los entrenamientos (PowerShell)

Con el entorno `yolov12` (sección 3) activado, ejecuta los cuatro
entrenamientos — Nano y Small, cada uno con Base y Augmented:

```powershell
# Nano — dataset base
python train_yolo12.py `
    --data data\visdrone_base.yaml `
    --name nano_base `
    --model yolo12n.pt `
    --epochs 250 `
    --imgsz 640 `
    --batch 16 `
    --workers 8

# Nano — dataset aumentado (offline)
python train_yolo12.py `
    --data data\visdrone_augmented.yaml `
    --name nano_augmented `
    --model yolo12n.pt `
    --epochs 250 `
    --imgsz 640 `
    --batch 16 `
    --workers 8

# Small — dataset base
python train_yolo12.py `
    --data data\visdrone_base.yaml `
    --name small_base `
    --model yolo12s.pt `
    --epochs 250 `
    --imgsz 640 `
    --batch 16 `
    --workers 8

# Small — dataset aumentado (offline)
python train_yolo12.py `
    --data data\visdrone_augmented.yaml `
    --name small_augmented `
    --model yolo12s.pt `
    --epochs 250 `
    --imgsz 640 `
    --batch 16 `
    --workers 8
```

> **`--name` único por combinación**: el script usa `exist_ok=True` en
> `model.train()`, así que dos corridas con el mismo `--name` se sobrescriben
> entre sí en `runs/YOLOv12/<name>/` — de ahí `nano`/`small` como prefijo
> combinado con `_base`/`_augmented` como sufijo, en vez de reutilizar el
> mismo nombre para las cuatro corridas. En W&B no hay riesgo de sobrescritura (cada
> corrida crea un run nuevo aunque el nombre se repita); se mantienen
> nombres únicos igual, para que el dashboard quede ordenado.

Los resultados locales quedan en subcarpetas independientes dentro del mismo
proyecto: `runs/YOLOv12/nano_base/`, `runs/YOLOv12/nano_augmented/`,
`runs/YOLOv12/small_base/` y `runs/YOLOv12/small_augmented/`. En
W&B, las cuatro corridas caen en el mismo proyecto (`YOLOv12`), distinguidas
por nombre de corrida.

> **Resolución de imagen (bajada al estándar YOLO, 640)**: las imágenes
> fuente son 1280×720; `imgsz=640` es el default de
> `ultralytics/cfg/default.yaml` y el tamaño con el que estos modelos fueron
> ajustados originalmente (COCO). Se probó primero a `1280` (nativo) y luego
> a `960` (punto medio) buscando conservar más detalle para las personas
> pequeñas/lejanas de VisDrone, pero ambos tamaños salen bastante más
> pesados en memoria y velocidad con YOLOv12 (los bloques de atención
> `A2C2f` / Area Attention escalan peor con resolución que una CNN
> convencional) — con `640` se priorizó velocidad y estabilidad de
> entrenamiento sobre ese detalle extra. En modo `train`, ultralytics recibe
> `imgsz` como un único entero que define el lado largo del letterbox
> cuadrado; el lado corto se rellena (padding) en vez de recortarse o
> deformarse. **Confirmado en este dataset, a `imgsz=1280` con `yolo12l.pt`
> (Large)**: `batch=8` produce `CUDA OutOfMemoryError` en
> `TaskAlignedAssigner` (VisDrone tiene muchísimas cajas por imagen, lo que
> infla el tensor de costo de asignación) — el motivo original del recorte
> de alcance a Nano/Small (ver nota al inicio del documento). A `640px` hay
> mucho más margen de VRAM (~6.25× menos píxeles que a 1280px) y estos
> modelos son mucho más chicos que Large, así que `--batch` sube a `16` — no
> verificado todavía con Nano/Small en este dataset; si da `OOM`, se
> recomienda bajarlo o usar `--batch -1` (autobatch, deja que ultralytics
> mida la VRAM libre real) — el mismo valor se usa en las cuatro corridas.

> **"El entrenamiento es muy lento / no avanza" (causa raíz confirmada)**: si
> aparece `WARNING: CUDA OutOfMemoryError in TaskAlignedAssigner, using CPU`
> justo al arrancar la época 1, **esa es la causa** — no un cuelgue.
> Ultralytics atrapa el `OutOfMemoryError` en ese paso puntual y hace
> fallback silencioso a CPU (mueve los tensores GPU→CPU, calcula ahí, los
> regresa a GPU), **en cada iteración**, lo que hace que la GPU se vea casi
> al límite de uso pero el entrenamiento avance extremadamente lento. Esto
> fue justo lo que ocurrió con `batch=8` a `imgsz=1280` en este dataset — de
> ahí que se bajara primero el `imgsz` (960, luego 640) y se reajustara el
> `batch` en cada paso (ver nota anterior). Si el batch ya está bajo y sigue
> lento sin ese warning específico, entran en juego motivos normales de
> rendimiento: los bloques de atención de YOLOv12 escalan peor con
> resolución que una CNN normal, y sin FlashAttention (mensaje
> `"FlashAttention is not available on this device. Using
> scaled_dot_product_attention instead."`, normal en Windows) el fallback
> `scaled_dot_product_attention` es más lento — en una GPU Blackwell tan
> reciente como la RTX 5060 Ti, los kernels siguen madurando. Conviene
> revisar el `s/it` / `ETA` de la barra de progreso antes de asumir un
> cuelgue: si el ETA es de horas por época, es lento, no un cuelgue.
> `--workers` por defecto es `8` — en una CPU con varios núcleos, un valor
> bajo deja el preprocesamiento (mosaic + albumentations a 1280px) como
> cuello de botella; se recomienda ajustar `--workers` según los núcleos
> lógicos de la CPU (el script avisa si se pasa por encima). Si aparece
> `BrokenPipeError` / `EOFError` (multiprocessing en Windows), baja
> `--workers` a `0`.

### Otros tamaños de modelo (Medium, Large — no usados actualmente)

Fuera del alcance actual del proyecto (ver nota al inicio del documento). El
script no cambió — `--model`, `--data` y `--name` siguen siendo parámetros
de línea de comandos, así que retomar Medium o Large no requiere tocar
código, solo estos comandos de referencia:

```powershell
# Medium — dataset base
python train_yolo12.py `
    --data data\visdrone_base.yaml `
    --name medium_base `
    --model yolo12m.pt `
    --epochs 250 `
    --imgsz 640 `
    --batch 16 `
    --workers 8

# Medium — dataset aumentado (offline)
python train_yolo12.py `
    --data data\visdrone_augmented.yaml `
    --name medium_augmented `
    --model yolo12m.pt `
    --epochs 250 `
    --imgsz 640 `
    --batch 16 `
    --workers 8
```

> `yolo12l.pt` (Large) ya se probó y confirmó demasiado pesado para esta GPU
> con este dataset a `imgsz=1280` (ver nota de resolución más arriba); no se
> recomienda reintentarlo sin bajar `--batch` agresivamente o usar
> `--batch -1` (autobatch).

## 10. Métricas registradas en W&B

La integración nativa de ultralytics (`ultralytics/utils/callbacks/wb.py`,
activada vía `settings.update({"wandb": True})`) ya registra
automáticamente, por época:

- `metrics/precision(B)`, `metrics/recall(B)`
- `metrics/mAP50(B)`, `metrics/mAP50-95(B)`
- Pérdidas de entrenamiento: `train/box_loss`, `train/cls_loss`, `train/dfl_loss`
- Curvas Precision-Recall, F1-Confidence, Precision-Confidence,
  Recall-Confidence (una serie por clase; aquí solo `person`)
- Matriz de confusión y artefacto del mejor checkpoint (`best.pt`)

`train_yolo12.py` agrega un callback adicional (`on_fit_epoch_end`) que
registra las mismas métricas con nombres explícitos bajo el prefijo
`person/` para lectura directa en el dashboard:

- `person/precision`, `person/recall`, `person/f1_score`
- `person/mAP50`, `person/mAP50-95`
- `person/iou_at_0.5` (= `mAP50`: fracción de detecciones con IoU ≥ 0.5, la
  definición operativa de "IoU" a nivel de dataset en detección de objetos
  — no existe un IoU escalar único por época en detección, a diferencia de
  segmentación)
- `person/accuracy` (índice de Jaccard `TP / (TP + FP + FN)` derivado de la
  matriz de confusión 2×2 `person` vs. `background`; es la métrica más
  cercana a "accuracy" en detección de un solo objeto, ya que no existe
  accuracy de clasificación estándar cuando no hay negativos verdaderos
  explícitos por imagen)

## 11. Verificación de GPU (incluida en el script)

Antes de cada entrenamiento, `train_yolo12.py` imprime y valida:

```python
torch.cuda.is_available()
torch.version.cuda
torch.cuda.get_device_name(0)
torch.cuda.get_device_capability(0)
```

Si `torch.cuda.is_available()` es `False`, el script aborta antes de cargar
el dataset o inicializar W&B.

## 12. Resolución de problemas

Problemas reales encontrados y resueltos durante estas pruebas, en el orden
en que suelen aparecer. Cada fila tiene la explicación completa en la
sección indicada.

| Síntoma | Causa | Fix | Detalle |
|---|---|---|---|
| `pip install -r requirements-windows.txt` falla con `Building wheel for stringzilla ... Microsoft Visual C++ 14.0 or greater is required` | `albumentations` arrastra `albucore`→`stringzilla>=3.10.4`, que no publica wheel para Windows desde su serie 2.x | Instalar *Build Tools for Visual Studio* y marcar explícitamente el workload **"Desktop development with C++"** (el instalador base solo, sin ese workload, no basta) | sección 2 |
| `ValueError: API key must be 40 characters long, yours was 86` al iniciar el entrenamiento | `wandb.login(key=...)` valida el formato clásico de key personal (40 caracteres); las keys con prefijo (`wandb_v1_...`, de cuentas de servicio/organización) no lo cumplen aunque sean válidas | El script ya no llama a `wandb.login()`; exporta `WANDB_API_KEY` como variable de entorno y deja que `wandb.init()` autentique contra el backend real | sección 8 |
| `wandb.errors.UsageError: Invalid project name '...': cannot contain characters '/,\,#,?,%,:'` | El callback nativo de ultralytics derivaba el nombre de proyecto de W&B a partir de una ruta local de Windows (con `\` y `:`) | El script llama a `wandb.init(project=..., name=...)` con el nombre de proyecto limpio antes de `model.train()` | sección 8 |
| GPU casi al 100% de uso pero el entrenamiento no avanza (época 1 pegada) | `WARNING: CUDA OutOfMemoryError in TaskAlignedAssigner, using CPU` — `batch`/`imgsz` demasiado altos para la VRAM disponible con este dataset (VisDrone tiene muchísimas cajas por imagen) | Bajar `--batch` y/o `--imgsz`, o usar `--batch -1` (autobatch); también se cambió el modelo de `yolo12l.pt` a `yolo12m.pt` | sección 9 (nota de imgsz/batch), historial de commits |
| Entrenamiento lento pero **sin** ese warning de OOM | Normal a mayor resolución con YOLOv12: los bloques de atención (`A2C2f`) escalan peor que una CNN, y sin FlashAttention (`"Using scaled_dot_product_attention instead"`, esperado en Windows) el fallback es más lento — más notorio en una GPU Blackwell reciente con kernels aún inmaduros | Revisar el `s/it`/`ETA` de la barra de progreso antes de asumir un cuelgue; bajar `imgsz` o subir `--workers` si el cuello de botella es el preprocesamiento en CPU | sección 9 |
| `torch.cuda.is_available()` da `True` pero el entrenamiento falla o cae a CPU sin avisar | El wheel de PyTorch instalado (`cu124`) no incluye kernels para Blackwell (`sm_120`, RTX 50-series) | Reinstalar con `--index-url https://download.pytorch.org/whl/cu128` | sección 1, sección 4 |
