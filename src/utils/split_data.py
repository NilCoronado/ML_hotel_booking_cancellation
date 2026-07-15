"""
Split conjunto train/test del dataset hotel_bookings.
Random_state y test_size fijados en equipo para que EDA (Persona A)
y modelado (Persona B) trabajen sobre el mismo split.

Nota: el CSV bruto no se versiona en el repo (para respetar el limite de
100MB en src/data_sample/). Descargar manualmente desde
https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand
y colocarlo en DATA_PATH antes de ejecutar este script. train.csv y
test.csv (ya generados) si estan versionados.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET = "is_canceled"

DATA_PATH = "src/data_sample/hotel_bookings.csv"
TRAIN_PATH = "src/data_sample/train.csv"
TEST_PATH = "src/data_sample/test.csv"

df = pd.read_csv(DATA_PATH)

train_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df[TARGET],
)

train_df.to_csv(TRAIN_PATH, index=False)
test_df.to_csv(TEST_PATH, index=False)

print(f"train: {train_df.shape}, test: {test_df.shape}")
print(f"proporcion cancelados train: {train_df[TARGET].mean():.4f}")
print(f"proporcion cancelados test: {test_df[TARGET].mean():.4f}")
