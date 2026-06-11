# ============================================
# IMPORTS
# ============================================

import os
import random

import pandas as pd

from PIL import Image

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision import transforms


# ============================================
# CONFIGURACIÓN
# ============================================

SEED = 43

random.seed(SEED)
torch.manual_seed(SEED)

DATASET_DIR = "data"

# Carpeta de imágenes (se usa PokemonDataNoBG si existe, sino PokemonData)
IMAGES_SUBDIR = "PokemonData"
if os.path.exists(os.path.join(DATASET_DIR, "PokemonDataNoBG")):
    IMAGES_SUBDIR = "PokemonDataNoBG"
    print(f"INFO: Usando imágenes preprocesadas en {IMAGES_SUBDIR}")
else:
    print(f"INFO: Usando imágenes originales en {IMAGES_SUBDIR}")

CSV_PATH = os.path.join(
    DATASET_DIR,
    "pokemon_dataset.csv"
)


# ============================================
# CARGAR CSV
# ============================================

df = pd.read_csv(CSV_PATH)

print("Cantidad total de imágenes:")
print(len(df))

print("\nPrimeras filas:")
print(df.head())


# ============================================
# CREAR ETIQUETAS NUMÉRICAS
# ============================================

classes = sorted(
    df["type1"].unique()
)

label_to_idx = {
    label: idx
    for idx, label in enumerate(classes)
}

idx_to_label = {
    idx: label
    for label, idx in label_to_idx.items()
}

df["label"] = df["type1"].map(
    label_to_idx
)

print("\nClases:")
print(label_to_idx)


# ============================================
# TRANSFORMACIONES
# ============================================

train_transform = transforms.Compose([

    transforms.Resize(256),

    transforms.CenterCrop(224),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(15),

    transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.5,
        hue=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_test_transform = transforms.Compose([

    transforms.Resize(256),

    transforms.CenterCrop(224),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================
# PARTICIÓN ESTRATIFICADA
# ============================================

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["label"],
    random_state=SEED
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=SEED
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print("\nCantidad de imágenes por conjunto")
print(f"Train: {len(train_df)}")
print(f"Validation: {len(val_df)}")
print(f"Test: {len(test_df)}")


# ============================================
# DATASET PERSONALIZADO
# ============================================

class PokemonDataset(Dataset):

    def __init__(
        self,
        dataframe,
        root_dir,
        transform=None
    ):

        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):

        return len(self.dataframe)

    def __getitem__(self, idx):

        row = self.dataframe.iloc[idx]

        # Si usamos imágenes sin fondo, intentamos usar el path sin fondo
        original_path = os.path.join(self.root_dir, row["image_path"])
        
        if IMAGES_SUBDIR == "PokemonDataNoBG":
            nobg_path = row["image_path"].replace("PokemonData", "PokemonDataNoBG")
            nobg_path = os.path.splitext(nobg_path)[0] + ".png"
            full_nobg_path = os.path.join(self.root_dir, nobg_path)
            
            # Si el archivo sin fondo existe, lo usamos. Si no, usamos el original.
            if os.path.exists(full_nobg_path):
                image_path = full_nobg_path
            else:
                image_path = original_path
        else:
            image_path = original_path

        image = Image.open(
            image_path
        ).convert("RGB")

        label = int(
            row["label"]
        )

        if self.transform:

            image = self.transform(
                image
            )

        return image, label


# ============================================
# CREAR DATASETS
# ============================================

train_dataset = PokemonDataset(
    train_df,
    DATASET_DIR,
    train_transform
)

val_dataset = PokemonDataset(
    val_df,
    DATASET_DIR,
    val_test_transform
)

test_dataset = PokemonDataset(
    test_df,
    DATASET_DIR,
    val_test_transform
)


# ============================================
# DATALOADERS
# ============================================

BATCH_SIZE = 32

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================
# VERIFICACIÓN
# ============================================

images, labels = next(
    iter(train_loader)
)

print("\nShape batch:")
print(images.shape)

print("\nShape labels:")
print(labels.shape)

print("\nRango de valores:")
print(images.min().item())
print(images.max().item())

