"""
Lab Sheet-08 | Project 1: Handwritten Digit Recognition using Neural Networks
MCA III Semester (Session 2026-2027)

Pipeline
  1. Load MNIST (falls back to scikit-learn's 8x8 digits if MNIST cannot be downloaded)
  2. Train a Deep Neural Network (Keras) - baseline vs. tuned hyperparameters
  3. Plot learning curves, confusion matrix and sample predictions
  4. Save the trained model
  5. Compare against traditional ML (Logistic Regression, Random Forest)

Run:  python mnist_dnn.py
"""
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import keras
from keras import layers

SEED = 42
np.random.seed(SEED)
keras.utils.set_random_seed(SEED)
OUT = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------------
# 1. Data
# ----------------------------------------------------------------------------
def load_data():
    """Return x_train, y_train, x_test, y_test (scaled 0-1) and dataset name."""
    try:
        from keras.datasets import mnist
        (x_tr, y_tr), (x_te, y_te) = mnist.load_data()
        name = "MNIST (28x28, 70,000 images)"
    except Exception as exc:  # no internet / blocked download
        print(f"[!] MNIST download failed ({type(exc).__name__}). "
              "Falling back to sklearn digits (8x8, 1,797 images).")
        from sklearn.datasets import load_digits
        d = load_digits()
        x = d.images.astype("float32") / 16.0 * 255.0
        x_tr, x_te, y_tr, y_te = train_test_split(
            x, d.target, test_size=0.25, stratify=d.target, random_state=SEED)
        name = "sklearn digits fallback (8x8, 1,797 images)"
    return (x_tr.astype("float32") / 255.0, y_tr,
            x_te.astype("float32") / 255.0, y_te, name)


x_train, y_train, x_test, y_test, DATASET = load_data()
IMG_SHAPE = x_train.shape[1:]
print(f"Dataset : {DATASET}")
print(f"Train   : {x_train.shape}  Test: {x_test.shape}")

# hold out 10% of the training data for validation
x_tr, x_val, y_tr, y_val = train_test_split(
    x_train, y_train, test_size=0.1, stratify=y_train, random_state=SEED)

# ----------------------------------------------------------------------------
# 2. Model builder + experiments
# ----------------------------------------------------------------------------
def build_model(hidden=(256, 128), activation="relu", dropout=0.2, lr=1e-3):
    model = keras.Sequential([layers.Input(shape=IMG_SHAPE), layers.Flatten()])
    for units in hidden:
        model.add(layers.Dense(units, activation=activation))
        model.add(layers.Dropout(dropout))
    model.add(layers.Dense(10, activation="softmax"))
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


EXPERIMENTS = {
    "Baseline (128, relu, lr=1e-3)": dict(hidden=(128,), activation="relu", dropout=0.0, lr=1e-3),
    "Deep (256-128, relu, drop=.2)": dict(hidden=(256, 128), activation="relu", dropout=0.2, lr=1e-3),
    "Deep (256-128, tanh, drop=.2)": dict(hidden=(256, 128), activation="tanh", dropout=0.2, lr=1e-3),
    "Deep (256-128, relu, lr=1e-2)": dict(hidden=(256, 128), activation="relu", dropout=0.2, lr=1e-2),
}
EPOCHS = 20 if x_train.shape[0] > 5000 else 60
BATCH = 128 if x_train.shape[0] > 5000 else 32

results, histories, models = [], {}, {}
for name, cfg in EXPERIMENTS.items():
    t0 = time.time()
    m = build_model(**cfg)
    h = m.fit(x_tr, y_tr, validation_data=(x_val, y_val), epochs=EPOCHS,
              batch_size=BATCH, verbose=0,
              callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
    loss, acc = m.evaluate(x_test, y_test, verbose=0)
    results.append(dict(Model=name, Test_Accuracy=acc, Test_Loss=loss,
                        Epochs_Run=len(h.history["loss"]), Train_Time_s=time.time() - t0))
    histories[name], models[name] = h.history, m
    print(f"{name:34s} acc={acc:.4f} loss={loss:.4f}")

res_df = pd.DataFrame(results).sort_values("Test_Accuracy", ascending=False)
best_name = res_df.iloc[0]["Model"]
best_model = models[best_name]
print("\nBest DNN:", best_name)

# ----------------------------------------------------------------------------
# 3. Plots
# ----------------------------------------------------------------------------
sns.set_theme(style="whitegrid")

# learning curves for every experiment
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
for name, h in histories.items():
    ax[0].plot(h["val_accuracy"], label=name)
    ax[1].plot(h["val_loss"], label=name)
ax[0].set(title="Validation accuracy per epoch", xlabel="Epoch", ylabel="Accuracy")
ax[1].set(title="Validation loss per epoch", xlabel="Epoch", ylabel="Loss")
ax[0].legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT}/learning_curves_all.png", dpi=150)
plt.close()

# best model: train vs validation
h = histories[best_name]
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(h["accuracy"], label="train"); ax[0].plot(h["val_accuracy"], label="validation")
ax[0].set(title=f"Accuracy - {best_name}", xlabel="Epoch"); ax[0].legend()
ax[1].plot(h["loss"], label="train"); ax[1].plot(h["val_loss"], label="validation")
ax[1].set(title="Loss", xlabel="Epoch"); ax[1].legend()
plt.tight_layout()
plt.savefig(f"{OUT}/learning_curves_best.png", dpi=150)
plt.close()

# confusion matrix
y_pred = best_model.predict(x_test, verbose=0).argmax(axis=1)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion matrix (best DNN, test set)")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{OUT}/confusion_matrix.png", dpi=150)
plt.close()

# sample predictions (green = correct, red = wrong)
idx = np.random.choice(len(x_test), 15, replace=False)
fig, axes = plt.subplots(3, 5, figsize=(10, 6.5))
for a, i in zip(axes.ravel(), idx):
    a.imshow(x_test[i], cmap="gray")
    ok = y_pred[i] == y_test[i]
    a.set_title(f"pred {y_pred[i]} / true {y_test[i]}", color="green" if ok else "red", fontsize=9)
    a.axis("off")
plt.tight_layout()
plt.savefig(f"{OUT}/sample_predictions.png", dpi=150)
plt.close()

# ----------------------------------------------------------------------------
# 4. Save model
# ----------------------------------------------------------------------------
best_model.save(f"{OUT}/mnist_dnn_model.keras")
print("Saved model -> mnist_dnn_model.keras")

# ----------------------------------------------------------------------------
# 5. Compare with traditional ML
# ----------------------------------------------------------------------------
flat_tr, flat_te = x_train.reshape(len(x_train), -1), x_test.reshape(len(x_test), -1)
# classic models are slow on full MNIST -> subsample the training data for them
n_ml = min(len(flat_tr), 10000)
sub = np.random.choice(len(flat_tr), n_ml, replace=False)
ml_models = {
    "Logistic Regression": LogisticRegression(max_iter=300),
    "Random Forest": RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=SEED),
}
rows = [dict(Model=f"DNN - {best_name}", Test_Accuracy=res_df.iloc[0]["Test_Accuracy"])]
for name, clf in ml_models.items():
    clf.fit(flat_tr[sub], y_train[sub])
    rows.append(dict(Model=name, Test_Accuracy=accuracy_score(y_test, clf.predict(flat_te))))
cmp_df = pd.DataFrame(rows)
cmp_df.to_csv(f"{OUT}/model_comparison.csv", index=False)
res_df.to_csv(f"{OUT}/hyperparameter_results.csv", index=False)

plt.figure(figsize=(8, 4))
sns.barplot(data=cmp_df, y="Model", x="Test_Accuracy", hue="Model", legend=False)
plt.xlim(cmp_df.Test_Accuracy.min() - 0.03, 1.0)
for i, v in enumerate(cmp_df.Test_Accuracy):
    plt.text(v + 0.001, i, f"{v:.3f}", va="center")
plt.title("Traditional ML vs Deep Learning")
plt.tight_layout()
plt.savefig(f"{OUT}/ml_vs_dl.png", dpi=150)
plt.close()

report = classification_report(y_test, y_pred, digits=4)
with open(f"{OUT}/classification_report.txt", "w") as f:
    f.write(f"Dataset: {DATASET}\nBest model: {best_name}\n\n{report}")

print("\n=== Hyperparameter experiments ===")
print(res_df.round(4).to_string(index=False))
print("\n=== ML vs DL ===")
print(cmp_df.round(4).to_string(index=False))
print("\n" + report)
