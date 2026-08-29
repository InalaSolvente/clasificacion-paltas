# Clasificación del estado de madurez de paltas

Framework modular para analizar y clasificar imágenes del **Avocado Ripening
Dataset**. La variable objetivo es `Ripening Index Classification`, una escala
ordinal de 1 a 5.

## EDA

```powershell
python -m pip install -r requirements.txt
python scripts/run_eda.py
```

El comando crea `reports/eda/EDA_REPORT.md`, un manifiesto auditable y los
gráficos. El manifiesto usa `sample_id` como identificador de la unidad
experimental: futuros splits deben agrupar por esta columna para evitar fuga de
datos entre fotos de la misma palta.

## Notebook de experimentos

```powershell
python -m pip install -r requirements-notebook.txt
jupyter lab notebooks/01_experiment_framework.ipynb
```

El notebook funciona como orquestador: la configuración de cada corrida está en
una sola celda y la lógica de datos, splits y registro vive en `src/avocado/`.
Las particiones 70/10/20 se realizan por palta y se estratifican por grupo de
almacenamiento. El notebook incluye una comparación reproducible de ResNet18,
EfficientNet-B0 y DINO ViT-S/16; el conjunto de test permanece bloqueado durante
la selección del modelo.

### CUDA en Windows

La RTX 5060 Ti requiere un build de PyTorch con soporte Blackwell. Instala CUDA
13 en el mismo kernel que ejecutará el notebook:

```powershell
python -m pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements-notebook.txt
python scripts/check_cuda.py
```

El entrenamiento tiene `accelerator="cuda"` y aborta si CUDA no está disponible;
no existe fallback silencioso a CPU. También habilita AMP y TF32.

### Validación y testing

El notebook ofrece dos flujos:

1. **Iteración simple:** entrena las arquitecturas, elige el mayor macro-F1 de
   validación y evalúa en test solamente ese checkpoint.
2. **Cross-validation:** mantiene fijo el 20% de test y ejecuta cinco folds por
   `sample_id` sobre el 80% de desarrollo. La arquitectura se selecciona por el
   macro-F1 medio de los folds y luego se ejecuta una única corrida final.

Los lados `a/b` y todos los días de una misma palta permanecen juntos tanto en
el split fijo como en cada fold.

El perfil para RTX 5060 Ti 16 GB compara ResNet50, EfficientNetV2-S,
ConvNeXt-Small, Swin-Tiny, DINO ViT-S/16 y DINO ViT-B/16. Usa AMP y un batch
efectivo de 64 mediante micro-batches adaptados y acumulación de gradientes. Al finalizar se genera
`experiments/cv_loss_evolution.png` con las curvas medias de training y
validation loss y la desviación entre folds.

El notebook también incluye un baseline interpretable de regresión logística
sobre descriptores de color RGB/HSV, textura y forma calculados sobre una
segmentación aproximada. Las augmentations neuronales se configuran en una celda
inmediatamente posterior a la carga de datos y se previsualizan antes de entrenar.

## Datos y artefactos no versionados

El repositorio conserva el código, notebooks, informes, configuraciones,
historiales y gráficos. Los siguientes artefactos se excluyen de Git por tamaño
y reproducibilidad:

- `Avocado Ripening Dataset/` y `Avocado Ripening Dataset.xlsx`;
- checkpoints `experiments/**/*.pt`;
- caché de features `experiments/handcrafted_features.csv`.

Para reproducir el proyecto, coloca el dataset y su Excel en la raíz con esos
nombres y ejecuta primero `python scripts/run_eda.py`. Los checkpoints se vuelven
a generar al ejecutar el notebook.
