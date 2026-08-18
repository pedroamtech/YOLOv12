"""Entrenamiento de YOLOv12-Large sobre VisDrone (clase única "person") en Windows.

Uso:
    python train_yolo12.py --data data/visdrone_base.yaml      --name visdrone_base
    python train_yolo12.py --data data/visdrone_augmented.yaml --name visdrone_augmented

Este script deliberadamente NO expone ni sobreescribe hiperparámetros de red
(lr0, optimizer, mosaic, mixup, fliplr, etc.): ambos experimentos deben usar
los valores por defecto de ultralytics/cfg/default.yaml para que la única
variable entre Experimento 1 y 2 sea el dataset (ver data/*.yaml).
"""

import argparse
import multiprocessing
import os
from pathlib import Path

import torch
import wandb
from dotenv import load_dotenv

from ultralytics import YOLO, settings

REPO_ROOT = Path(__file__).resolve().parent


def load_credentials() -> dict:
    """Carga variables de entorno desde .env y valida que exista la API key de W&B."""
    load_dotenv(REPO_ROOT / ".env")
    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        raise SystemExit(
            "WANDB_API_KEY no está definida. Copia .env.example a .env y "
            "rellena tu clave (https://wandb.ai/authorize)."
        )
    return {
        "api_key": api_key,
        "project": os.getenv("WANDB_PROJECT", "YOLOv12-VisDrone"),
        "entity": os.getenv("WANDB_ENTITY") or None,
    }


def verify_gpu() -> None:
    """Verifica disponibilidad de CUDA y muestra la GPU detectada antes de entrenar."""
    print("=" * 70)
    print("Verificación de entorno CUDA / PyTorch")
    print("=" * 70)
    print(f"PyTorch version   : {torch.__version__}")
    print(f"CUDA disponible   : {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA no disponible. Verifica el driver NVIDIA y que instalaste el "
            "wheel de PyTorch con soporte CUDA (ver requirements-windows.txt)."
        )
    print(f"CUDA version (torch): {torch.version.cuda}")
    print(f"GPU detectada     : {torch.cuda.get_device_name(0)}")
    print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM total (GB)   : {vram_gb:.1f}")
    print("=" * 70)


def make_wandb_metric_logger():
    """Callback on_fit_epoch_end: registra métricas con nombres explícitos para la clase 'person'."""

    def log_custom_wandb_metrics(trainer):
        if wandb.run is None:
            return

        m = trainer.metrics
        precision = float(m.get("metrics/precision(B)", 0.0))
        recall = float(m.get("metrics/recall(B)", 0.0))
        map50 = float(m.get("metrics/mAP50(B)", 0.0))
        map50_95 = float(m.get("metrics/mAP50-95(B)", 0.0))
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        log_dict = {
            "person/precision": precision,
            "person/recall": recall,
            "person/f1_score": f1,
            "person/mAP50": map50,
            "person/mAP50-95": map50_95,
            "person/iou_at_0.5": map50,  # mAP@IoU=0.50: fracción de detecciones correctas con IoU>=0.5
        }

        # Accuracy tipo Jaccard (TP / (TP+FP+FN)) a partir de la matriz de confusión
        # 2x2 (person vs. background) del validador de esta época.
        cm = getattr(trainer.validator, "confusion_matrix", None)
        if cm is not None and getattr(cm, "matrix", None) is not None and cm.matrix.shape == (2, 2):
            tp, fp, fn = cm.matrix[0, 0], cm.matrix[0, 1], cm.matrix[1, 0]
            denom = tp + fp + fn
            if denom > 0:
                log_dict["person/accuracy"] = float(tp / denom)

        wandb.run.log(log_dict, step=trainer.epoch + 1)

    return log_custom_wandb_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena YOLOv12-Large sobre VisDrone en Windows")
    parser.add_argument("--data", required=True, help="Ruta al .yaml del dataset (data/visdrone_base.yaml o data/visdrone_augmented.yaml)")
    parser.add_argument("--name", required=True, help="Nombre de la corrida (subcarpeta de resultados y run de W&B)")
    parser.add_argument("--model", default="yolo12l.pt", help="Pesos/config del modelo (default: yolo12l.pt)")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--imgsz", type=int, default=1280, help="Imágenes fuente en 720p (1280x720): imgsz=1280 evita reescalar/perder detalle en el lado largo")
    parser.add_argument("--batch", type=int, default=8, help="16 GB VRAM: 8 es un punto de partida seguro para yolo12l @1280px (Area Attention escala más que una CNN pura); usa -1 para autobatch")
    parser.add_argument("--workers", type=int, default=8, help="Workers de dataloader; sube/baja según núcleos de CPU disponibles, usa 0 si aparecen BrokenPipeError/EOFError en Windows")
    parser.add_argument("--patience", type=int, default=50)
    args = parser.parse_args()
    if os.name == "nt" and args.workers > 0:
        cpu_count = os.cpu_count() or 1
        if args.workers > cpu_count:
            print(f"Aviso: workers={args.workers} supera los {cpu_count} hilos lógicos detectados; considera bajarlo.")
    return args


def main() -> None:
    creds = load_credentials()
    verify_gpu()

    # No se usa wandb.login(key=...): esa función escribe la key en ~/.netrc y
    # valida que tenga exactamente 40 caracteres (formato clásico de API key
    # personal), lo que rompe con keys más nuevas con prefijo (p. ej.
    # "wandb_v1_...", de cuentas de servicio/organización). Exportar la
    # variable de entorno evita esa validación: wandb.init() (llamado
    # internamente por el callback nativo de ultralytics) la toma directo y
    # autentica contra el backend real, que es la validación que importa.
    os.environ["WANDB_API_KEY"] = creds["api_key"]
    if creds["entity"]:
        os.environ["WANDB_ENTITY"] = creds["entity"]
    settings.update({"wandb": True})  # habilita la integración nativa de ultralytics con W&B

    args = parse_args()

    # Un único proyecto de W&B (${WANDB_PROJECT}, ya existente en el
    # dashboard); Base vs. Augmented se distingue por el NOMBRE de la corrida
    # (--name), no por el proyecto.
    #
    # Se inicializa W&B nosotros mismos, ANTES de model.train(). El callback
    # nativo de ultralytics (on_pretrain_routine_start en wb.py) solo llama a
    # wb.init(...) si todavía no hay un run activo; si lo dejáramos derivar el
    # nombre de proyecto él solo, usa el `project=` que le pasamos a
    # model.train() (una ruta de carpeta local, ver abajo) y solo limpia '/',
    # no '\' ni ':' — en Windows eso revienta la validación de nombre de
    # proyecto de W&B. Con el run ya abierto aquí, el callback nativo detecta
    # wb.run y solo loguea métricas en él, sin volver a inicializar nada.
    wandb.init(project=creds["project"], name=args.name, entity=creds["entity"])

    model = YOLO(args.model)
    model.add_callback("on_fit_epoch_end", make_wandb_metric_logger())

    # Solo parámetros de ejecución/hardware. NO se pasan overrides de lr0, optimizer,
    # mosaic, mixup, fliplr, etc.: ambos experimentos heredan exactamente los mismos
    # valores de ultralytics/cfg/default.yaml.
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=0,
        amp=True,
        patience=args.patience,
        project=str(REPO_ROOT / "runs" / creds["project"]),
        name=args.name,
        exist_ok=True,
        seed=0,
        plots=True,
        verbose=True,
    )

    print(f"Entrenamiento finalizado. Resultados en: runs/{creds['project']}/{args.name}")


if __name__ == "__main__":
    multiprocessing.freeze_support()  # requerido en Windows para multiprocessing con dataloader workers>0
    main()
