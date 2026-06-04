# Información de los Datasets

Esta carpeta contiene los notebooks y la documentación relacionada con la preparación de datos para el proyecto de Clasificación de Tipos de Pokemon.

## Datasets Utilizados

### 1. Datos Tabulares de Pokemon
- **URL:** [https://www.kaggle.com/datasets/mlomuscio/pokemon](https://www.kaggle.com/datasets/mlomuscio/pokemon)
- **Descripción:** Este dataset contiene un archivo CSV (`PokemonData.csv`) con información de todas las generaciones de Pokemon, incluyendo sus nombres, tipos, generaciones y otras estadísticas. Lo usamos principalmente para obtener los tipos correctos de cada Pokemon.

### 2. Datos de Imágenes de Pokemon
- **URL:** [https://www.kaggle.com/datasets/lantian773030/pokemonclassification](https://www.kaggle.com/datasets/lantian773030/pokemonclassification)
- **Descripción:** Este dataset contiene imágenes de varios Pokemon organizadas en carpetas por nombre. Se utiliza como fuente para nuestro modelo de clasificación de imágenes.

## Proceso de Preparación de Datos

La preparación de los datos se maneja en el notebook `01-preparacion-dataset.ipynb`, que realiza los siguientes pasos:
1. Descarga los datasets de Kaggle.
2. Filtra los datos tabulares para los Pokemon de la Generación 1.
3. Empareja las imágenes con sus tipos correspondientes del CSV.
4. Realiza la limpieza y normalización de los datos.
5. Divide los datos en conjuntos de entrenamiento, validación y prueba.
6. Aplica transformaciones de imagen y prepara los DataLoaders para PyTorch.
