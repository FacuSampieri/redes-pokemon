# Producción - App Streamlit

Esta carpeta contiene la aplicación web del TP Integrador.

## Archivos

- `app.py`: interfaz Streamlit y lógica de interacción del juego.
- `utils.py`: carga del modelo, preprocesamiento, predicción y consulta de Pokémon.
- `requirements.txt`: dependencias con versiones fijadas.

## Modelo

La app carga los pesos finales desde:

```text
../dev/modelo.pth
```

El preprocesamiento de producción es el mismo usado en validación/test durante el entrenamiento:

- resize a `224x224`;
- conversión a tensor;
- normalización ImageNet con `mean=[0.485, 0.456, 0.406]` y `std=[0.229, 0.224, 0.225]`.

## Ejecución local

Desde la raíz del repositorio:

```bash
pip install -r prod/requirements.txt
streamlit run prod/app.py
```

## Uso

1. La app genera un lote de 20 Pokémon de primera generación.
2. El usuario elige el tipo principal de cada Pokémon.
3. La IA predice el tipo usando la imagen oficial.
4. La app compara usuario vs IA y muestra puntajes, aciertos, errores y probabilidades.

## Despliegue

URL pública:

```text
PENDIENTE: agregar URL cuando esté desplegada.
```
