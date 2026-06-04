# Clasificación de Tipos de Pokemon con Redes Neuronales

Este proyecto consiste en el desarrollo de una red neuronal profunda diseñada para clasificar imágenes de Pokemon de la primera generación según su tipo principal (Fuego, Agua, Planta, etc.).

## Objetivo

El objetivo principal es aplicar técnicas de aprendizaje profundo (Deep Learning) y visión por computadora para identificar patrones visuales que caracterizan a los diferentes tipos de Pokemon, utilizando la arquitectura de PyTorch.

## Estructura del Proyecto

- `data/`: Contiene los scripts y notebooks para la descarga y preparación del dataset.
  - `01-preparacion-dataset.ipynb`: Notebook principal para la descarga y preprocesamiento de imágenes.
  - `readme.md`: Información detallada sobre las fuentes de datos.
- `requirements.txt`: Lista de dependencias necesarias para ejecutar el proyecto.

## Dataset

Se utilizan dos fuentes principales de Kaggle:
1. **Información Tabular:** Para obtener las etiquetas (tipos) correctas de cada Pokemon.
2. **Imágenes:** Un conjunto de datos con miles de imágenes de Pokemon organizadas por nombre.

## Requisitos

Para instalar las dependencias necesarias, ejecuta:

```bash
pip install -r requirements.txt
```

## Cómo empezar

1. Dirígete a la carpeta `data/`.
2. Ejecuta el notebook `01-preparacion-dataset.ipynb` para preparar los datos.
3. (Próximamente) Entrenamiento del modelo en la carpeta `dev/`.
