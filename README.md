# Laboratorio web de datos sintéticos

Aplicación Gradio para **entrenar** y luego **generar** datos con TVAE, CTAB-GAN+ y Tabula desde Google Colab.

## Origen de cada modelo

| Modelo | Implementación utilizada |
|---|---|
| TVAE | `sdv.single_table.TVAESynthesizer`, igual que el notebook `TESIS_Final TVAE.ipynb` |
| CTAB-GAN+ | Código original incluido en `vendor/ctabgan_model/`, proveniente de `CTAB-GAN-Plus/model/` |
| Tabula | Código original incluido en `vendor/tabula_middle_padding/`, proveniente de `Tabula/tabula_middle_padding/` |

No se incluye el dataset real ni modelos entrenados en GitHub. Se cargan mediante la interfaz y los resultados quedan en `runs/` durante la sesión. En Colab conviene configurar `SYNTHETIC_RUNS_DIR` en Google Drive para conservarlos.

## Uso en Colab

1. Abra `Ejecutar_en_Google_Colab.ipynb` desde GitHub.
2. Seleccione una GPU en **Entorno de ejecución > Cambiar tipo de entorno de ejecución**.
3. Ejecute las celdas en orden.
4. Abra el enlace público `gradio.live`.
5. En **Entrenar modelo**, cargue el CSV, seleccione el modelo y ajuste sus parámetros.
6. Descargue el modelo entrenado.
7. En **Generar datos**, cargue ese modelo, indique la cantidad de filas y genere sin reentrenar.

## Consideraciones

- CTAB-GAN+ requiere indicar correctamente sus columnas categóricas, enteras, generales, logarítmicas y mixtas.
- Tabula descarga el modelo base de Hugging Face la primera vez y puede requerir bastante memoria GPU.
- La barra de Gradio informa las fases. TVAE y Tabula además entregan una curva de pérdida cuando su implementación expone el historial.
- El historial registra parámetros, duración, estado, artefactos y observaciones.
