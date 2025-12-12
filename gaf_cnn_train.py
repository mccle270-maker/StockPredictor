import numpy as np
import pandas as pd
from pyts.image import GramianAngularField
from tensorflow import keras
from tensorflow.keras import layers

from data_fetch import get_history_cached  # reuse your code
import time

from tensorflow import keras
gaf_cnn = keras.models.load_model("gaf_cnn_updown.keras")

# Config
tickers = ["AAPL"]          # keep small for now
window = 30
image_size = 30
horizon = 1                 # 1‑day up/down
max_samples = 100           # hard cap so it can't explode

gaf = GramianAngularField(image_size=image_size, method="summation")

input_shape = (image_size, image_size, 1)

model = keras.Sequential([
    layers.Input(shape=input_shape),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)  # [web:712][web:793]

if __name__ == "__main__":
    print("Building GAF dataset...")
    print(f"Tickers: {tickers}, window={window}, horizon={horizon}")

    X_imgs = []
    y_labels = []

    for tk in tickers:
        hist = get_history_cached(tk, period="3y", interval="1d")
        close = hist["Close"].dropna()
        rets = close.pct_change().dropna().values

        for i in range(window, len(rets) - horizon):
            if len(X_imgs) >= max_samples:
                break

            window_vals = rets[i - window:i]
            future_ret = rets[i + horizon - 1]
            y = 1 if future_ret > 0 else 0

            X = window_vals.reshape(1, -1)
            img = gaf.fit_transform(X)[0]  # (image_size, image_size) [web:394][web:783]
            X_imgs.append(img)
            y_labels.append(y)

    X = np.array(X_imgs, dtype="float32")[..., np.newaxis]
    y = np.array(y_labels, dtype="int32")

    print("Dataset shape:", X.shape, y.shape)

    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    print("Starting training...")
    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=1,        # just 1 epoch
        batch_size=16,
        verbose=2,       # one line per epoch
    )
    print("Training time (s):", time.time() - t0)

    model.save("gaf_cnn_updown.keras")
    print("Saved model to gaf_cnn_updown.keras")
