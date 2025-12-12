import numpy as np
import pandas as pd
from pyts.image import GramianAngularField
from tensorflow import keras
from tensorflow.keras import layers

from data_fetch import get_history_cached  # reuse your code

tickers = ["AAPL"]  # start small
window = 30
image_size = 30
horizon = 1     # 1-day up/down

gaf = GramianAngularField(image_size=image_size, method="summation")

X_imgs = []
y_labels = []

for tk in tickers:
    hist = get_history_cached(tk, period="5y", interval="1d")
    close = hist["Close"].dropna()
    rets = close.pct_change().dropna().values

    # rolling windows
    for i in range(window, len(rets) - horizon):
        window_vals = rets[i-window:i]
        future_ret = rets[i + horizon - 1]   # horizon-ahead daily return
        y = 1 if future_ret > 0 else 0

        X = window_vals.reshape(1, -1)
        img = gaf.fit_transform(X)[0]   # shape (image_size, image_size)

        X_imgs.append(img)
        y_labels.append(y)

X = np.array(X_imgs, dtype="float32")[..., np.newaxis]  # (N, H, W, 1)
y = np.array(y_labels, dtype="int32")

# simple train/val split
split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

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
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=2,
    batch_size=64,
    verbose=2,
)

if __name__ == "__main__":
    print("Building GAF dataset...")
    print(f"Tickers: {tickers}, window={window}, horizon={horizon}")

    X_imgs = []
    y_labels = []

    for tk in tickers:
        hist = get_history_cached(tk, period="5y", interval="1d")
        close = hist["Close"].dropna()
        rets = close.pct_change().dropna().values

        for i in range(window, len(rets) - horizon):
            window_vals = rets[i-window:i]
            future_ret = rets[i + horizon - 1]
            y = 1 if future_ret > 0 else 0

            X = window_vals.reshape(1, -1)
            img = gaf.fit_transform(X)[0]
            X_imgs.append(img)
            y_labels.append(y)

    X = np.array(X_imgs, dtype="float32")[..., np.newaxis]
    y = np.array(y_labels, dtype="int32")

    print("Dataset shape:", X.shape, y.shape)

    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=64,
    )

    model.save("gaf_cnn_updown.keras")
    print("Saved model to gaf_cnn_updown.keras")




