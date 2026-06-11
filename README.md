# Clasificación de Tipos Pokémon mediante Redes Neuronales

## Descripción

Este proyecto tiene como objetivo desarrollar un sistema de clasificación automática de tipos Pokémon utilizando técnicas de Deep Learning y Transfer Learning sobre modelos preentrenados.

La propuesta consiste en construir un pipeline completo que abarque:

Preparación y procesamiento del dataset.
Aplicación de técnicas de data augmentation.
Particionado reproducible de los datos.
Entrenamiento de redes neuronales convolucionales preentrenadas.
Fine-tuning de modelos seleccionados.
Evaluación y comparación de resultados mediante métricas de clasificación.

El dataset utilizado contiene imágenes de Pokémon de la primera generación asociadas a su tipo principal, permitiendo abordar un problema de clasificación multiclase.

Actualmente el repositorio contiene:

- Dataset final procesado.
- Imágenes organizadas por Pokémon.
- `dataset_loader.py`: Script de carga y preparación de datos para PyTorch. Detecta automáticamente si se procesaron las imágenes (sin fondo) y las utiliza.
- `dev/01_remove_background.ipynb`: Notebook para eliminar el fondo de las imágenes (Opcional pero recomendado para Entrega 3).
- `dev/02_model_training.ipynb`: Notebook con el entrenamiento, experimentación y evaluación de modelos (Core de la Entrega 3).

---

## Estructura del proyecto

```text
.
├── data/
│   ├── pokemon_dataset.csv
│   └── PokemonData/              # Imágenes originales
│   └── PokemonDataNoBG/          # Imágenes sin fondo (generadas por el notebook 01)
│
├── dev/
│   ├── 01_remove_background.ipynb
│   └── 02_model_training.ipynb
│
├── dataset_loader.py
├── requirements.txt
└── README.md
```

---

## Requisitos

- Python 3.10 o superior
- Conexión a internet para instalar dependencias

---

## Crear entorno virtual

Desde la raíz del proyecto:

```bash
python3 -m venv .venv
```

Activar el entorno virtual:

```bash
source .venv/bin/activate
```

---

## Instalar dependencias

Con el entorno virtual activado:

```bash
pip install -r requirements.txt
```

---

## Ejecutar el script

Desde la raíz del proyecto:

```bash
python dataset_loader.py
```

El script realizará las siguientes tareas:

- Carga del dataset desde el archivo CSV.
- Creación de etiquetas numéricas para cada tipo Pokémon.
- Definición de transformaciones para entrenamiento y evaluación.
- División estratificada del dataset en:
  - 70% entrenamiento
  - 15% validación
  - 15% prueba
- Creación de los objetos Dataset de PyTorch.
- Creación de los DataLoader correspondientes.
- Verificación de dimensiones y rangos de los tensores generados.

---

## Desactivar el entorno virtual

Una vez finalizado el trabajo:

```bash
deactivate
```

---

## Tecnologías utilizadas

- Python
- PyTorch
- Torchvision
- Pandas
- Scikit-learn
- Pillow
