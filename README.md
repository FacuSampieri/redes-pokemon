# Clasificador de tipo principal de Pokémon

Trabajo Práctico Integrador de la materia **Redes Neuronales Profundas**.

El proyecto entrena una red neuronal de visión por computadora para clasificar imágenes de Pokémon de primera generación según su **tipo principal**. Además, incluye una aplicación web en Streamlit donde el usuario compite contra el modelo intentando adivinar el tipo de 20 Pokémon.

## Integrantes

- Facundo Sampieri
- Fabrizio Avallone
- Julieta Vicente
- Tomas Guiñazu
- Martin Manrrique

## Problema elegido

El problema abordado es de **clasificación multiclase de imágenes**. A partir de una imagen de un Pokémon, el modelo debe predecir su tipo principal: `Bicho`, `Dragon`, `Electrico`, `Fuego`, `Agua`, etc.

Se eligió una red neuronal porque la clasificación depende de patrones visuales como forma, color, textura, silueta y variabilidad entre imágenes. Estos patrones son difíciles de resolver correctamente con reglas manuales simples.

## Dataset

Se utilizan dos datasets públicos de Kaggle:

1. **Pokemon with stats** (`mlomuscio/pokemon`): contiene información tabular de los Pokémon, incluyendo generación y tipo principal.
2. **Pokemon Image Dataset** (`lantian773030/pokemonclassification`): contiene imágenes organizadas por nombre de Pokémon.

En este trabajo se filtran los Pokémon de **primera generación** para mantener un alcance manejable. La etiqueta usada es el tipo principal, traducido al mismo conjunto de clases que usa la aplicación.

Más detalle sobre descarga y estructura: [`data/README.md`](data/README.md).

## Modelo

Se realiza **fine-tuning** de modelos preentrenados de `torchvision`. La arquitectura preentrenada se modifica reemplazando la capa final para que la salida coincida con la cantidad de clases del problema.

Estrategias contempladas en el notebook:

- ResNet18 con entrenamiento solo de la capa final.
- ResNet18 descongelando `layer4` y la capa final.
- ResNet34 descongelando `layer4` y la capa final.

La función de pérdida utilizada es `CrossEntropyLoss` ponderada por clase para reducir el impacto del desbalance del dataset.

## Aplicación web

La app está en `prod/` y fue integrada con el modelo entrenado. Permite jugar un lote de 20 Pokémon: el usuario elige el tipo de cada uno y luego la aplicación compara sus respuestas contra las predicciones de la IA.

La interfaz muestra:

- puntaje del usuario;
- puntaje de la IA;
- predicción del modelo;
- tipo real;
- probabilidades por clase;
- resumen de aciertos y errores.

URL de la app desplegada:

```text
PENDIENTE: agregar URL pública cuando esté desplegada.
```

## Estructura del repositorio

```text
.
├── data/
│   ├── README.md
│   ├── train.csv              # se genera al correr el notebook
│   ├── val.csv                # se genera al correr el notebook
│   └── test.csv               # se genera al correr el notebook
├── dev/
│   ├── 01_dataset_training_pokemon.ipynb
│   ├── modelo.pth
│   └── model_metadata.json    # se genera al correr el notebook
├── prod/
│   ├── app.py
│   ├── utils.py
│   ├── requirements.txt
│   └── README.md
├── .gitignore
└── README.md
```

Notas:

- Las imágenes del dataset no se suben al repositorio.
- `data/raw/` se genera localmente al ejecutar el notebook y queda ignorado por Git.
- Los CSV `train.csv`, `val.csv` y `test.csv` sí deben versionarse porque son livianos y garantizan que todos usen el mismo particionado.
- El archivo `dev/modelo.pth` contiene los pesos finales del modelo usado por la app.

## Cómo ejecutar el proyecto en local

### 1. Clonar el repositorio

```bash
git clone <URL_DEL_REPOSITORIO>
cd <NOMBRE_DEL_REPOSITORIO>
```

### 2. Crear un entorno virtual

```bash
python -m venv .venv
source .venv/bin/activate  # Linux / macOS
# .venv\Scripts\activate   # Windows
```

### 3. Instalar dependencias

```bash
pip install -r prod/requirements.txt
```

### 4. Ejecutar el notebook de desarrollo

Abrir y correr:

```text
dev/01_dataset_training_pokemon.ipynb
```

El notebook realiza los pasos principales:

1. descarga automática de los datasets;
2. limpieza y mapeo de imágenes con etiquetas;
3. análisis de distribución por clase;
4. generación de `train.csv`, `val.csv` y `test.csv`;
5. definición de `Dataset` y `DataLoader`;
6. visualización de augmentations y batches;
7. entrenamiento de tres configuraciones de fine-tuning;
8. evaluación final sobre test;
9. guardado de `dev/modelo.pth` y `dev/model_metadata.json`.

### 5. Ejecutar la app

Desde la raíz del repo:

```bash
streamlit run prod/app.py
```

La aplicación carga los pesos desde:

```text
dev/modelo.pth
```

## Métricas reportadas

El notebook reporta:

- accuracy de validación por experimento;
- curvas de loss y accuracy;
- accuracy final sobre test;
- precision, recall y F1 por clase;
- matriz de confusión;
- ejemplos mal clasificados para análisis de errores.

## Reproducibilidad

Para reproducir el entrenamiento desde una máquina nueva:

1. clonar el repositorio;
2. instalar dependencias;
3. ejecutar el notebook desde cero;
4. verificar que se generen `data/train.csv`, `data/val.csv`, `data/test.csv`, `dev/modelo.pth` y `dev/model_metadata.json`.

Se fija una semilla (`SEED = 251`) para mantener la partición reproducible.

## Observaciones para la defensa oral

Puntos importantes para explicar:

- Por qué el problema es adecuado para una red neuronal.
- Por qué se eligió fine-tuning en lugar de entrenamiento desde cero.
- Cómo se reemplazó la capa final del modelo preentrenado.
- Cómo se manejó el desbalance de clases.
- Por qué las augmentations solo se aplican en entrenamiento.
- Por qué la métrica final se reporta sobre test y no sobre validación.
- Cómo la app usa el mismo preprocesamiento de validación/test.
- Qué errores comete el modelo y posibles causas.
