# Dataset

Esta carpeta contiene la documentación y los archivos livianos del dataset.

## Fuentes

Se usan dos datasets públicos de Kaggle:

1. `mlomuscio/pokemon`: datos tabulares de Pokémon, generación y tipo principal.
2. `lantian773030/pokemonclassification`: imágenes organizadas por nombre de Pokémon.

## Descarga

La descarga se realiza automáticamente desde el notebook:

```text
dev/01_dataset_training_pokemon.ipynb
```

El notebook usa `kagglehub` y copia los datasets dentro de:

```text
data/raw/
```

La carpeta `data/raw/` no se versiona porque contiene imágenes y archivos pesados.

## Splits

Al ejecutar el notebook se generan estos CSV livianos:

```text
data/train.csv
data/val.csv
data/test.csv
```

Cada CSV contiene:

- nombre del Pokémon;
- tipo principal traducido al formato de la app;
- ruta relativa de la imagen;
- tipo original en inglés.

Estos CSV sí deben versionarse para que todos los integrantes usen exactamente el mismo particionado.

## Particionado

El particionado se hace de forma estratificada con semilla fija:

- 70% entrenamiento;
- 15% validación;
- 15% test.

La semilla usada en el notebook es:

```text
SEED = 251
```
