"""Interfaz Gradio para entrenar y generar datos con TVAE, CTAB-GAN+ y Tabula.

Las implementaciones de CTAB-GAN+ y Tabula provienen de los códigos originales
entregados para el Capstone. TVAE utiliza la implementación oficial de SDV.
"""
from __future__ import annotations

import json
import os
import pickle
import random
import shutil
import tempfile
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = Path(os.getenv("SYNTHETIC_RUNS_DIR", BASE_DIR / "runs"))
RUNS_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_PATH = RUNS_DIR / "historial.jsonl"


def fix_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_csv(path: str) -> pd.DataFrame:
    if not path:
        raise gr.Error("Primero cargue un archivo CSV.")
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


def names(text: str) -> list[str]:
    return [value.strip() for value in (text or "").split(",") if value.strip()]


def validate_columns(data: pd.DataFrame, *groups: list[str]) -> None:
    requested = {item for group in groups for item in group}
    missing = sorted(requested - set(data.columns))
    if missing:
        raise gr.Error(f"Estas columnas no existen en el CSV: {', '.join(missing)}")


def run_dir(model: str, operation: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RUNS_DIR / f"{stamp}_{model.lower().replace('+', 'plus')}_{operation}_{uuid.uuid4().hex[:6]}"
    path.mkdir(parents=True)
    return path


def record(model: str, operation: str, status: str, seconds: float, params: dict,
           artifact: str = "", notes: str = "") -> None:
    row = {
        "fecha_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "modelo": model,
        "operacion": operation,
        "estado": status,
        "duracion_segundos": round(seconds, 3),
        "parametros": params,
        "artefacto": artifact,
        "observaciones": notes or "",
    }
    with HISTORY_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def history_frame() -> pd.DataFrame:
    columns = ["fecha_utc", "modelo", "operacion", "estado", "duracion_segundos", "parametros", "artefacto", "observaciones"]
    if not HISTORY_PATH.exists():
        return pd.DataFrame(columns=columns)
    rows = [json.loads(line) for line in HISTORY_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["parametros"] = frame["parametros"].map(lambda value: json.dumps(value, ensure_ascii=False))
    return frame.reindex(columns=columns).iloc[::-1]


def zip_folder(folder: Path, destination: Path) -> Path:
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in folder.rglob("*"):
            if path.is_file() and path != destination:
                archive.write(path, path.relative_to(folder))
    return destination


def save_loss_plot(values: pd.DataFrame, path: Path, x: str, y: str, title: str) -> str | None:
    if values is None or values.empty or x not in values or y not in values:
        return None
    clean = values.dropna(subset=[x, y])
    if clean.empty:
        return None
    plt.figure(figsize=(9, 4.5))
    plt.plot(clean[x], clean[y], color="#147D92")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.grid(alpha=.25)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
    return str(path)


def train(model_name, csv_path, epochs, batch_size, embedding_dim, seed,
          categorical, integer, general, log_columns, mixed_json, test_ratio,
          base_model, conditional_index, notes, progress=gr.Progress(track_tqdm=True)):
    started = time.perf_counter()
    model_name = str(model_name)
    params = {"epochs": int(epochs), "batch_size": int(batch_size), "seed": int(seed)}
    out = run_dir(model_name, "entrenamiento")
    try:
        progress(0.03, desc="Leyendo y validando el CSV")
        data = read_csv(csv_path)
        fix_seed(int(seed))
        if data.empty:
            raise gr.Error("El CSV no contiene registros.")
        progress(0.10, desc=f"Preparando {model_name}")

        if model_name == "TVAE":
            from sdv.metadata import Metadata
            from sdv.single_table import TVAESynthesizer
            metadata = Metadata.detect_from_dataframe(data=data, table_name="datos")
            synth = TVAESynthesizer(metadata=metadata, epochs=int(epochs), batch_size=int(batch_size),
                                    embedding_dim=int(embedding_dim), enable_gpu=torch.cuda.is_available(), verbose=True)
            progress(0.18, desc="Entrenando TVAE")
            synth.fit(data)
            model_path = out / "modelo_tvae.pkl"
            synth.save(filepath=model_path)
            loss = synth.get_loss_values()
            if not loss.empty:
                loss = loss.groupby("Epoch", as_index=False)["Loss"].mean()
            plot_path = save_loss_plot(loss, out / "curva_perdida.png", "Epoch", "Loss", "Pérdida de TVAE")
            params["embedding_dim"] = int(embedding_dim)
            artifact = model_path

        elif model_name == "CTAB-GAN+":
            from vendor.ctabgan_model.ctabgan import CTABGAN
            cats, ints, generals, logs = names(categorical), names(integer), names(general), names(log_columns)
            mixed = json.loads(mixed_json or "{}")
            validate_columns(data, cats, ints, generals, logs, list(mixed))
            source_csv = out / "datos_entrenamiento.csv"
            data.to_csv(source_csv, index=False)
            progress(0.18, desc="Entrenando CTAB-GAN+")
            synth = CTABGAN(raw_csv_path=str(source_csv), test_ratio=float(test_ratio),
                            categorical_columns=cats, log_columns=logs, mixed_columns=mixed,
                            general_columns=generals, non_categorical_columns=[], integer_columns=ints,
                            problem_type={None: None}, batch_size=int(batch_size), epochs=int(epochs))
            synth.fit()
            artifact = out / "modelo_ctabgan_plus.pkl"
            with artifact.open("wb") as handle:
                pickle.dump(synth, handle)
            plot_path = None
            params.update({"categorical_columns": cats, "integer_columns": ints, "general_columns": generals,
                           "log_columns": logs, "mixed_columns": mixed, "test_ratio": float(test_ratio)})

        else:
            from vendor.tabula_middle_padding import Tabula
            columns = data.columns.tolist()
            num_cols = data.select_dtypes(include="number").columns.tolist()
            index = int(conditional_index)
            if not 0 <= index < len(columns):
                raise gr.Error(f"El índice condicional debe estar entre 0 y {len(columns)-1}.")
            model_dir = out / "modelo_tabula"
            progress(0.18, desc="Entrenando Tabula")
            synth = Tabula(llm=str(base_model), experiment_dir=str(model_dir), batch_size=int(batch_size),
                           epochs=int(epochs), columns=columns, num_cols=num_cols)
            trainer = synth.fit(data, conditional_col=columns[index], column_names=columns)
            synth.save(str(model_dir))
            history = pd.DataFrame(trainer.state.log_history)
            plot_path = save_loss_plot(history, out / "curva_perdida.png", "step", "loss", "Pérdida de Tabula")
            artifact = zip_folder(model_dir, out / "modelo_tabula.zip")
            params.update({"base_model": str(base_model), "conditional_column_index": index})

        elapsed = time.perf_counter() - started
        manifest = {"model": model_name, "created_utc": datetime.now(timezone.utc).isoformat(),
                    "rows": len(data), "columns": list(data.columns), "params": params}
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        record(model_name, "entrenamiento", "OK", elapsed, params, str(artifact), notes)
        progress(1, desc="Entrenamiento terminado")
        return f"Modelo entrenado en {elapsed:.1f} s. Registros usados: {len(data):,}.", str(artifact), plot_path, history_frame()
    except Exception as exc:
        record(model_name, "entrenamiento", "ERROR", time.perf_counter() - started, params, "", f"{notes} | {exc}")
        raise gr.Error(f"No se pudo entrenar {model_name}: {exc}") from exc


def unpack_uploaded_model(model_name: str, uploaded_path: str, destination: Path):
    if not uploaded_path:
        raise gr.Error("Seleccione el modelo previamente entrenado.")
    path = Path(uploaded_path)
    if model_name == "Tabula":
        if path.suffix.lower() != ".zip":
            raise gr.Error("Para Tabula debe cargar modelo_tabula.zip.")
        model_dir = destination / "modelo_tabula"
        model_dir.mkdir()
        with zipfile.ZipFile(path) as archive:
            archive.extractall(model_dir)
        return model_dir
    return path


def generate(model_name, trained_model, rows, max_length, temperature, notes,
             progress=gr.Progress(track_tqdm=True)):
    started = time.perf_counter()
    model_name, rows = str(model_name), int(rows)
    params = {"filas_sinteticas": rows}
    out = run_dir(model_name, "generacion")
    try:
        if rows <= 0:
            raise gr.Error("La cantidad de filas debe ser mayor que cero.")
        progress(.08, desc="Cargando el modelo entrenado")
        source = unpack_uploaded_model(model_name, trained_model, out)
        if model_name == "TVAE":
            from sdv.utils import load_synthesizer
            synth = load_synthesizer(filepath=source)
            progress(.25, desc=f"Generando {rows:,} filas con TVAE")
            synthetic = synth.sample(num_rows=rows)
        elif model_name == "CTAB-GAN+":
            with Path(source).open("rb") as handle:
                synth = pickle.load(handle)
            progress(.25, desc=f"Generando {rows:,} filas con CTAB-GAN+")
            synthetic = synth.generate_samples(n=rows)
        else:
            from vendor.tabula_middle_padding import Tabula
            synth = Tabula.load_from_dir(str(source))
            progress(.25, desc=f"Generando {rows:,} filas con Tabula")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            synthetic = synth.sample(n_samples=rows, max_length=int(max_length), device=device,
                                     temperature=float(temperature))
            params.update({"max_length": int(max_length), "temperature": float(temperature), "device": device})
        result = out / f"datos_sinteticos_{model_name.lower().replace('+', 'plus').replace('-', '_')}.csv"
        synthetic.to_csv(result, index=False)
        elapsed = time.perf_counter() - started
        record(model_name, "generacion", "OK", elapsed, params, str(result), notes)
        progress(1, desc="Generación terminada")
        return f"Generación terminada en {elapsed:.1f} s. Filas creadas: {len(synthetic):,}.", synthetic.head(20), str(result), history_frame()
    except Exception as exc:
        record(model_name, "generacion", "ERROR", time.perf_counter() - started, params, "", f"{notes} | {exc}")
        raise gr.Error(f"No se pudieron generar los datos con {model_name}: {exc}") from exc


CSS = """
.gradio-container {max-width: 1220px !important;} .title {text-align:center; color:#12355b}
.note {background:#eef7fa; border-left:5px solid #14b8a6; padding:12px}
"""


with gr.Blocks(title="Laboratorio de datos sintéticos", css=CSS) as demo:
    gr.Markdown("# Laboratorio de datos sintéticos", elem_classes="title")
    gr.Markdown("Entrene y reutilice **TVAE, CTAB-GAN+ o Tabula**. El entrenamiento y la generación son procesos separados.", elem_classes="note")
    with gr.Tab("1. Entrenar modelo"):
        with gr.Row():
            with gr.Column(scale=1):
                train_model = gr.Dropdown(["TVAE", "CTAB-GAN+", "Tabula"], value="TVAE", label="Modelo")
                dataset = gr.File(label="Dataset real (.csv)", file_types=[".csv"], type="filepath")
                epochs = gr.Number(value=300, precision=0, minimum=1, label="Épocas")
                batch = gr.Dropdown([8, 16, 32, 64, 128, 256, 500], value=32, label="Batch size")
                seed = gr.Number(value=42, precision=0, label="Semilla")
                notes_train = gr.Textbox(label="Observaciones de la ejecución", lines=2)
            with gr.Column(scale=1):
                gr.Markdown("### Parámetros específicos")
                embedding = gr.Number(value=128, precision=0, label="TVAE · embedding dimension")
                categorical = gr.Textbox(label="CTAB-GAN+ · columnas categóricas (separadas por coma)")
                integer = gr.Textbox(label="CTAB-GAN+ · columnas enteras (separadas por coma)")
                general = gr.Textbox(label="CTAB-GAN+ · columnas generales (separadas por coma)")
                logs = gr.Textbox(label="CTAB-GAN+ · columnas logarítmicas (separadas por coma)")
                mixed = gr.Textbox(value="{}", label='CTAB-GAN+ · columnas mixtas (JSON; ej. {"col":[0.0]})')
                ratio = gr.Slider(0, .5, value=.2, step=.05, label="CTAB-GAN+ · test ratio")
                base_model = gr.Textbox(value="distilgpt2", label="Tabula · modelo base")
                conditional = gr.Number(value=0, precision=0, minimum=0, label="Tabula · índice de columna condicional")
        train_button = gr.Button("Entrenar y guardar modelo", variant="primary")
        train_status = gr.Textbox(label="Estado", interactive=False)
        with gr.Row():
            trained_download = gr.File(label="Descargar modelo entrenado")
            loss_plot = gr.Image(label="Avance/pérdida del entrenamiento")

    with gr.Tab("2. Generar datos"):
        gr.Markdown("Use un modelo ya entrenado. **Esta acción no vuelve a entrenarlo.**")
        with gr.Row():
            gen_model = gr.Dropdown(["TVAE", "CTAB-GAN+", "Tabula"], value="TVAE", label="Modelo")
            uploaded_model = gr.File(label="Modelo entrenado (.pkl o .zip)", type="filepath")
            n_rows = gr.Number(value=14359, precision=0, minimum=1, label="Cantidad de filas sintéticas")
        with gr.Row():
            max_length = gr.Number(value=400, precision=0, minimum=1, label="Tabula · longitud máxima")
            temperature = gr.Slider(.1, 2, value=1, step=.1, label="Tabula · temperatura")
            notes_gen = gr.Textbox(label="Observaciones", lines=2)
        generate_button = gr.Button("Generar datos sintéticos", variant="primary")
        gen_status = gr.Textbox(label="Estado", interactive=False)
        preview = gr.Dataframe(label="Vista previa (20 filas)", interactive=False)
        synthetic_download = gr.File(label="Descargar CSV sintético")

    with gr.Tab("3. Historial y observaciones"):
        refresh = gr.Button("Actualizar historial")
        history = gr.Dataframe(value=history_frame, label="Registro de ejecuciones", interactive=False)
        history_file = gr.File(value=lambda: str(HISTORY_PATH) if HISTORY_PATH.exists() else None,
                               label="Descargar historial JSONL")

    train_button.click(train, [train_model, dataset, epochs, batch, embedding, seed, categorical, integer,
                               general, logs, mixed, ratio, base_model, conditional, notes_train],
                       [train_status, trained_download, loss_plot, history])
    generate_button.click(generate, [gen_model, uploaded_model, n_rows, max_length, temperature, notes_gen],
                          [gen_status, preview, synthetic_download, history])
    refresh.click(history_frame, outputs=history).then(lambda: str(HISTORY_PATH) if HISTORY_PATH.exists() else None,
                                                       outputs=history_file)


if __name__ == "__main__":
    demo.queue().launch(share=True, debug=True)
