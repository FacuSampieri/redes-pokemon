import json
import os
import random
from io import BytesIO
from pathlib import Path

import requests
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

TRADUCCION_TIPOS = {
    "Bug": "Bicho",
    "Dragon": "Dragon",
    "Electric": "Electrico",
    "Fairy": "Hada",
    "Fighting": "Lucha",
    "Fire": "Fuego",
    "Ghost": "Fantasma",
    "Grass": "Planta",
    "Ground": "Tierra",
    "Ice": "Hielo",
    "Normal": "Normal",
    "Poison": "Veneno",
    "Psychic": "Psiquico",
    "Rock": "Roca",
    "Water": "Agua",
}

# Orden de clases usado por el entrenamiento. Debe coincidir con el orden guardado en el notebook.
DEFAULT_CLASSES = list(TRADUCCION_TIPOS.values())
CLASSES = DEFAULT_CLASSES.copy()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def obtener_transformacion():
    """Devuelve el mismo preprocesamiento usado en validación/test durante el entrenamiento."""
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def cargar_clases_desde_metadata(ruta_pesos):
    """Intenta leer dev/model_metadata.json para reconstruir el orden real de clases."""
    ruta_pesos = Path(ruta_pesos)
    posibles = [
        ruta_pesos.with_name("model_metadata.json"),
        ruta_pesos.parent / "model_metadata.json",
        Path(__file__).resolve().parents[1] / "dev" / "model_metadata.json",
    ]
    for metadata_path in posibles:
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            clases = metadata.get("classes")
            if isinstance(clases, list) and clases:
                return clases
    return DEFAULT_CLASSES.copy()


def cargar_modelo_ganador(ruta_pesos):
    """Carga una ResNet18 con la capa final adaptada y pesos entrenados por el grupo."""
    global CLASSES
    ruta_pesos = Path(ruta_pesos)

    if not ruta_pesos.exists():
        return None, f"No se encontraron los pesos en: {ruta_pesos}", False

    checkpoint = torch.load(ruta_pesos, map_location=DEVICE)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

    # Detectar cantidad de clases desde los pesos para evitar desajustes silenciosos.
    fc_weight = state_dict.get("fc.weight")
    if fc_weight is None:
        return None, "El archivo de pesos no corresponde a una ResNet18 con capa fc.", False

    num_classes = fc_weight.shape[0]
    clases_metadata = cargar_clases_desde_metadata(ruta_pesos)
    if len(clases_metadata) == num_classes:
        CLASSES = clases_metadata
    else:
        CLASSES = DEFAULT_CLASSES[:num_classes]

    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    model.load_state_dict(state_dict)

    epoch = checkpoint.get("epoch", "N/A") if isinstance(checkpoint, dict) else "N/A"
    model.to(DEVICE)
    model.eval()
    return model, f"OK: pesos cargados desde {ruta_pesos} (epoca {epoch})", True


def predecir_imagen(modelo, imagen_pil):
    """Procesa una imagen PIL y retorna índice predicho, probabilidades y clases."""
    transform = obtener_transformacion()
    imagen_tensor = transform(imagen_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = modelo(imagen_tensor)
        probabilidades = torch.nn.functional.softmax(outputs[0], dim=0)
        pred_idx = outputs.argmax(dim=1).item()

    return pred_idx, probabilidades.cpu(), CLASSES


def obtener_pokemon_por_id(id_pokemon):
    """Trae un Pokémon de primera generación, su imagen oficial y su tipo principal."""
    url = f"https://pokeapi.co/api/v2/pokemon/{id_pokemon}"

    try:
        respuesta = requests.get(url, timeout=20).json()
        nombre = respuesta["name"].capitalize()
        tipo_ingles = respuesta["types"][0]["type"]["name"].capitalize()
        tipo_real = TRADUCCION_TIPOS.get(tipo_ingles, "Normal")
        url_imagen = respuesta["sprites"]["other"]["official-artwork"]["front_default"]

        res_img = requests.get(url_imagen, timeout=20)
        imagen_pil = Image.open(BytesIO(res_img.content)).convert("RGB")

        return {
            "id": id_pokemon,
            "nombre": nombre,
            "tipo_real": tipo_real,
            "imagen": imagen_pil,
        }
    except Exception:
        return {
            "id": id_pokemon,
            "nombre": "Pikachu",
            "tipo_real": "Electrico",
            "imagen": Image.new("RGB", (224, 224), color="yellow"),
        }


def obtener_pokemon_aleatorio():
    """Trae un Pokémon aleatorio de primera generación."""
    return obtener_pokemon_por_id(random.randint(1, 151))


def obtener_lote_pokemon(cantidad=20):
    """Genera un lote de Pokémon únicos para jugar en modo masivo."""
    cantidad = max(1, min(cantidad, 20))
    ids = random.sample(range(1, 152), cantidad)
    return [obtener_pokemon_por_id(id_pokemon) for id_pokemon in ids]
