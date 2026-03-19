# ============================================================
# Stock Price Prediction using STFT + CNN
# Assignment 2 - Pattern Recognition for Financial Time Series
# ============================================================
# Companies : Reliance Industries (RELIANCE.NS)
#             Infosys            (INFY.NS)
#             Wipro              (WIPRO.NS)
#             TCS                (TCS.NS)
# ============================================================

# ── Imports ─────────────────────────────────────────────────
import os
import sys
import warnings
warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"
import tensorflow as tf
from tensorflow.keras import layers, models, Input, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import stft
from sklearn.preprocessing import MinMaxScaler

# ── Constants ───────────────────────────────────────────────
TICKERS      = ["RELIANCE.NS", "INFY.NS", "WIPRO.NS", "TCS.NS"]
START_DATE   = "2018-01-01"
END_DATE     = "2026-01-01"
NPERSEG      = 128
NOVERLAP     = 120
FS           = 1
PREDICT_STEP = 5
DATA_DIR     = "data/"
IMAGE_DIR    = "images/"
MODEL_DIR    = "models/"

for _d in [DATA_DIR, IMAGE_DIR, MODEL_DIR]:
    os.makedirs(_d, exist_ok=True)


# ════════════════════════════════════════════════════════════
# Task 1: Data Collection & Preprocessing
# ════════════════════════════════════════════════════════════
def collect_data():
    """Download and preprocess multivariate stock data for all tickers."""
    import yfinance as yf

    all_signals = {}

    print("\n  Downloading stock data...")
    for ticker in TICKERS:
        safe     = ticker.replace(".", "_")
        raw_path = os.path.join(DATA_DIR, f"{safe}_raw.csv")

        if os.path.exists(raw_path) and os.path.getsize(raw_path) > 500:
            print(f"  → {ticker}  (loading cached CSV)")
            signal = pd.read_csv(raw_path, index_col=0, parse_dates=True)
        else:
            print(f"  → Fetching {ticker} from Yahoo Finance ...")
            df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            if df.empty or len(df) < 50:
                print(f"    ⚠  No data returned for {ticker} — skipping")
                continue

            signal = pd.DataFrame(index=df.index)
            signal["Close"]    = df["Close"]
            signal["Volume"]   = df["Volume"]
            signal["HL_Range"] = df["High"] - df["Low"]
            signal["MA7"]      = df["Close"].rolling(7).mean()
            signal["MA30"]     = df["Close"].rolling(30).mean()
            signal.dropna(inplace=True)
            signal.to_csv(raw_path)
            print(f"    Saved → {raw_path}  ({len(signal)} rows, {signal.shape[1]} features)")

        all_signals[ticker] = signal

    if len(all_signals) < 2:
        raise RuntimeError("Not enough tickers downloaded. Check your internet connection.")

    close_combined = pd.concat(
        [all_signals[t]["Close"].rename(t) for t in all_signals], axis=1
    )
    before = len(close_combined)
    close_combined.dropna(inplace=True)
    common_dates = close_combined.index
    after = len(common_dates)
    print(f"\n  Common trading days: {after}  (dropped {before - after} rows)")

    for ticker in list(all_signals.keys()):
        all_signals[ticker] = all_signals[ticker].loc[common_dates]

    scalers           = {}
    normalized_signals = {}

    for ticker, sig in all_signals.items():
        scaler = MinMaxScaler()
        data_scaled = pd.DataFrame(
            scaler.fit_transform(sig),
            index=sig.index,
            columns=sig.columns
        )
        scalers[ticker]            = scaler
        normalized_signals[ticker] = data_scaled

    close_combined.to_csv(os.path.join(DATA_DIR, "combined_raw.csv"))
    close_combined_norm = pd.DataFrame(
        MinMaxScaler().fit_transform(close_combined),
        index=close_combined.index,
        columns=close_combined.columns
    )
    close_combined_norm.to_csv(os.path.join(DATA_DIR, "combined_normalized.csv"))
    print("  Saved → data/combined_raw.csv")
    print("  Saved → data/combined_normalized.csv")

    return normalized_signals, scalers


# ============================================================
# Task 2: Signal Processing - STFT + Spectrogram
# ============================================================
def generate_spectrograms(data):
    """Apply STFT on each feature of each stock."""
    spectrograms = {}

    print()
    for ticker, df in data.items():
        features     = df.columns.tolist()
        specs_linear = []
        specs_db     = []

        for feat in features:
            signal    = df[feat].values
            f, t, Zxx = stft(signal, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP)
            S_linear  = np.abs(Zxx) ** 2
            S_db      = 10 * np.log10(S_linear + 1e-9)
            specs_linear.append(S_linear)
            specs_db.append(S_db)

        stacked_linear = np.stack(specs_linear, axis=-1)
        stacked_db     = np.stack(specs_db,     axis=-1)

        spectrograms[ticker] = {
            "spec"     : stacked_db,
            "spec_raw" : stacked_linear,
            "f"        : f,
            "t"        : t,
            "features" : features
        }

        safe     = ticker.replace(".", "_")
        npy_path = os.path.join(DATA_DIR, f"{safe}_spectrogram.npy")
        np.save(npy_path, stacked_db)

        n_feat = len(features)
        fig, axes = plt.subplots(1, n_feat, figsize=(4 * n_feat, 4))
        if n_feat == 1:
            axes = [axes]
        fig.suptitle(f"Multi-Feature Spectrograms — {ticker}", fontsize=13, fontweight="bold")
        for idx, (feat, S_db) in enumerate(zip(features, specs_db)):
            ax   = axes[idx]
            mesh = ax.pcolormesh(t, f, S_db, shading="gouraud", cmap="inferno")
            ax.set_title(feat, fontsize=10)
            ax.set_xlabel("Time (days)")
            ax.set_ylabel("Freq (cycles/day)" if idx == 0 else "")
            plt.colorbar(mesh, ax=ax, label="dB")
        plt.tight_layout()
        plt.savefig(os.path.join(IMAGE_DIR, f"{safe}_multifeature_spec.png"), dpi=150)
        plt.close()

        fig, ax    = plt.subplots(figsize=(10, 4))
        S_close_db = specs_db[0]
        vmin, vmax = np.percentile(S_close_db, [5, 95])
        mesh = ax.pcolormesh(t, f, S_close_db, shading="gouraud",
                             cmap="inferno", vmin=vmin, vmax=vmax)
        ax.set_title(f"Spectrogram (Close - dB scale) - {ticker}", fontsize=13)
        ax.set_xlabel("Time (trading days)")
        ax.set_ylabel("Frequency (cycles/day)")
        plt.colorbar(mesh, ax=ax, label="Power (dB)")
        plt.tight_layout()
        plt.savefig(os.path.join(IMAGE_DIR, f"{safe}_spectrogram.png"), dpi=150)
        plt.close()

        print(f"  → {ticker}")
        print(f"     Signal length  : {len(df)} days")
        print(f"     STFT shape     : {stacked_db.shape}  (freq_bins × time_frames × features)")
        print(f"     Freq bins      : {len(f)}   Time frames: {len(t)}")

    return spectrograms


# ============================================================
# Task 3: Visualization
# ============================================================
def visualize(data, spectrograms):
    """Plot time series, frequency spectrum, and spectrograms."""
    print()

    fig, ax = plt.subplots(figsize=(12, 4))
    colors  = ["steelblue", "darkorange", "seagreen", "mediumpurple"]
    for i, ticker in enumerate(data):
        ax.plot(data[ticker].index, data[ticker]["Close"],
                label=ticker, linewidth=1.2, color=colors[i % len(colors)])
    ax.set_title("Normalized Close Prices — All Stocks", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized Price (0–1)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(IMAGE_DIR, "all_stocks_timeseries.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")

    for ticker in data:
        signal    = data[ticker]["Close"].values
        spec_data = spectrograms[ticker]
        S_db      = spec_data["spec"][:, :, 0]
        f_axis    = spec_data["f"]
        t_axis    = spec_data["t"]
        N         = len(signal)
        dates     = data[ticker].index

        fft_vals = np.abs(np.fft.rfft(signal))
        fft_freq = np.fft.rfftfreq(N, d=1)

        fig = plt.figure(figsize=(16, 5))
        gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)
        fig.suptitle(f"{ticker} - Signal Analysis", fontsize=14, fontweight="bold")

        ax0 = fig.add_subplot(gs[0])
        ax0.plot(dates, signal, color="steelblue", linewidth=1)
        ax0.set_title("Time Series  p(t)")
        ax0.set_xlabel("Date")
        ax0.set_ylabel("Normalized Price")
        ax0.grid(True, alpha=0.3)
        ax0.tick_params(axis="x", rotation=30)

        ax1      = fig.add_subplot(gs[1])
        dom_idx  = np.argmax(fft_vals[1:]) + 1
        dom_freq = fft_freq[dom_idx]
        ax1.plot(fft_freq, fft_vals, color="darkorange", linewidth=1)
        ax1.axvline(dom_freq, color="red", linestyle="--", linewidth=0.8,
                    label=f"Dominant: {dom_freq:.4f} cyc/day")
        ax1.set_title("Frequency Spectrum  |FFT|")
        ax1.set_xlabel("Frequency (cycles/day)")
        ax1.set_ylabel("Amplitude")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2        = fig.add_subplot(gs[2])
        vmin, vmax = np.percentile(S_db, [2, 98])
        mesh = ax2.pcolormesh(t_axis, f_axis, S_db, shading="gouraud",
                              cmap="inferno", vmin=vmin, vmax=vmax)
        ax2.set_title("Spectrogram  S(t, f)  [dB]")
        ax2.set_xlabel("Time (trading days)")
        ax2.set_ylabel("Frequency (cycles/day)")
        plt.colorbar(mesh, ax=ax2, label="Power (dB)")

        plt.tight_layout()
        safe = ticker.replace(".", "_")
        path = os.path.join(IMAGE_DIR, f"{safe}_analysis.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved → {path}  [{ticker}]")

    print("\n  All visualizations complete.")


# ============================================================
# Helper: Prepare CNN Dataset from Spectrograms
# ============================================================
def prepare_dataset(data, spectrograms):
    """
    Slice spectrograms into (X, y) pairs for CNN training.

    X[i] : spectrogram patch  (freq_bins, SPEC_WIN, n_features)
    y[i] : Close price PREDICT_STEP trading days ahead of the patch

    FIX 2 – label alignment:
        The STFT t-axis gives the centre-time (in samples) of each frame.
        We map frame index i → original signal index via t_axis[i].
        y is looked up at  round(t_axis[i]) + PREDICT_STEP  in the prices array.
    """
    X_all, y_all = [], []

    for ticker, df in data.items():
        stacked     = spectrograms[ticker]["spec"]   # (freq_bins, time_frames, features)
        t_axis      = spectrograms[ticker]["t"]       # centre-times in samples
        prices      = df["Close"].values              # normalized prices

        time_frames  = stacked.shape[1]

        # FIX 5 – guard SPEC_WIN against too-few frames
        SPEC_WIN = min(30, time_frames - PREDICT_STEP - 2)
        if SPEC_WIN < 5:
            print(f"  ⚠  {ticker}: too few STFT frames ({time_frames}), skipping")
            continue

        for i in range(SPEC_WIN, time_frames):
            # Spectrogram input patch
            patch = stacked[:, i - SPEC_WIN : i, :]           # (freq_bins, SPEC_WIN, features)

            # FIX 2 – correct label index via t_axis
            centre_sample = int(round(t_axis[i - 1]))         # end of patch in signal
            label_idx     = centre_sample + PREDICT_STEP

            if label_idx >= len(prices):
                break

            X_all.append(patch)
            y_all.append(prices[label_idx])

    if len(X_all) == 0:
        raise RuntimeError("Dataset is empty — STFT produced too few frames. "
                           "Lower NPERSEG or collect more data.")

    X = np.array(X_all, dtype=np.float32)   # (N, freq_bins, SPEC_WIN, features)
    y = np.array(y_all, dtype=np.float32)

    # Clip extreme dB values for CNN stability
    X = np.clip(X, -80, 0)
    # Re-normalize patches to [0, 1]  (dB range is negative; -80 is lowest)
    X = (X + 80) / 80

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"\n  Dataset prepared:")
    print(f"    Total samples : {len(X)}")
    print(f"    X shape       : {X.shape}  (samples, freq_bins, time_win, features)")
    print(f"    Train samples : {len(X_train)}")
    print(f"    Test  samples : {len(X_test)}")

    return X_train, X_test, y_train, y_test


# ============================================================
# Task 4: CNN Model - Improved Architecture
# ============================================================
def build_cnn_model(input_shape):
    """
    3-block CNN regression model with BatchNormalization and Dropout.

    Architecture:
        Input (freq_bins × time_win × features)
        ├─ Conv2D(32) + BN + ReLU → MaxPool
        ├─ Conv2D(64) + BN + ReLU → MaxPool
        ├─ Conv2D(128) + BN + ReLU → GlobalAvgPool
        ├─ Dense(256) + Dropout(0.4)
        ├─ Dense(64)  + Dropout(0.2)
        └─ Dense(1)   (linear — regression output)
    """
    reg = regularizers.l2(1e-4)

    model = models.Sequential([
        Input(shape=input_shape),

        # -- Block 1 ------------------------------------------
        layers.Conv2D(32, (3, 3), padding="same", kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),

        # -- Block 2 ------------------------------------------
        layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),

        # -- Block 3 ------------------------------------------
        layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.GlobalAveragePooling2D(),    # avoids flattening size issues

        # -- Fully Connected -----------------------------------
        layers.Dense(256, activation="relu", kernel_regularizer=reg),
        layers.Dropout(0.40),

        layers.Dense(64, activation="relu"),
        layers.Dropout(0.20),

        # -- Regression Output ---------------------------------
        layers.Dense(1)             # linear activation
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    print("\n  CNN model built successfully.")
    print(f"  Input shape  : {input_shape}")
    model.summary()
    return model


# ============================================================
# Task 5: Training
# ============================================================
def train_model(model, X_train, y_train):
    """Train the CNN with EarlyStopping, ModelCheckpoint, and ReduceLROnPlateau."""
    print()

    early_stop = EarlyStopping(
        monitor="val_loss", patience=15,
        restore_best_weights=True, verbose=1
    )

    checkpoint = ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, "cnn_stock_model.keras"),
        monitor="val_loss", save_best_only=True, verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=7,
        min_lr=1e-6, verbose=1
    )

    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop, checkpoint, reduce_lr],
        verbose=1
    )

    # -- Plot Training History ---------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Training History", fontsize=13, fontweight="bold")

    axes[0].plot(history.history["loss"],     label="Train Loss", color="steelblue")
    axes[0].plot(history.history["val_loss"], label="Val Loss",   color="darkorange")
    axes[0].set_title("Loss (MSE)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["mae"],     label="Train MAE", color="steelblue")
    axes[1].plot(history.history["val_mae"], label="Val MAE",   color="darkorange")
    axes[1].set_title("Mean Absolute Error")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(IMAGE_DIR, "training_history.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  Training history saved → {path}")

    return history


# ============================================================
# Task 6: Evaluation
# ============================================================
def evaluate_model(model, X_test, y_test):
    """Evaluate using MSE/RMSE/MAE and plot actual vs predicted."""

    print()

    y_pred = model.predict(X_test, verbose=0).flatten()

    mse  = float(np.mean((y_pred - y_test) ** 2))
    mae  = float(np.mean(np.abs(y_pred - y_test)))
    rmse = float(np.sqrt(mse))

    print(f"  -- Evaluation Results --------------------------")
    print(f"  MSE  (Mean Squared Error)      : {mse:.6f}")
    print(f"  RMSE (Root Mean Squared Error) : {rmse:.6f}")
    print(f"  MAE  (Mean Absolute Error)     : {mae:.6f}")
    print(f"  ------------------------------------------------")

    # -- Actual vs Predicted figure ----------------------------
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    fig.suptitle("CNN Stock Price Prediction - Evaluation", fontsize=14, fontweight="bold")

    axes[0].plot(y_test, label="Actual Price",    color="steelblue",  linewidth=1.2)
    axes[0].plot(y_pred, label="Predicted Price", color="darkorange",
                 linewidth=1.2, linestyle="--")
    axes[0].set_title("Actual vs Predicted (Normalized Price)")
    axes[0].set_xlabel("Sample Index")
    axes[0].set_ylabel("Normalized Price (0–1)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].text(0.02, 0.95,
                 f"MSE = {mse:.6f}   RMSE = {rmse:.6f}   MAE = {mae:.6f}",
                 transform=axes[0].transAxes, fontsize=9,
                 verticalalignment="top",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6))

    # Scatter plot — perfect = diagonal
    min_val = min(float(y_test.min()), float(y_pred.min()))
    max_val = max(float(y_test.max()), float(y_pred.max()))
    axes[1].scatter(y_test, y_pred, alpha=0.4, s=10, color="purple")
    axes[1].plot([min_val, max_val], [min_val, max_val],
                 "r--", linewidth=1.5, label="Perfect Prediction")
    axes[1].set_title("Scatter: Actual vs Predicted")
    axes[1].set_xlabel("Actual Price")
    axes[1].set_ylabel("Predicted Price")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(IMAGE_DIR, "actual_vs_predicted.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  Saved → {path}")

    # -- Save metrics ------------------------------------------
    metrics_path = os.path.join(DATA_DIR, "evaluation_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as fh:
        fh.write("=== CNN Stock Prediction - Evaluation Metrics ===\n\n")
        fh.write(f"MSE  : {mse:.6f}\n")
        fh.write(f"RMSE : {rmse:.6f}\n")
        fh.write(f"MAE  : {mae:.6f}\n")
        fh.write(f"\nTest samples : {len(y_test)}\n")
        fh.write(f"Tickers      : {', '.join(TICKERS)}\n")
        fh.write(f"Date range   : {START_DATE} -> {END_DATE}\n")
    print(f"  Saved → {metrics_path}")

    return {"mse": mse, "rmse": rmse, "mae": mae}


# ============================================================
# Bonus: Export CSVs -> Excel
# ============================================================
def export_to_excel():
    """Convert all raw CSVs in data/ to a single Excel file."""

    excel_path = os.path.join(DATA_DIR, "stock_data.xlsx")

    # FIX 4 – build sheet mapping dynamically from TICKERS (no hardcoded dots)
    csv_sheets = {}
    for ticker in TICKERS:
        safe = ticker.replace(".", "_")
        name = ticker.split(".")[0]                          # e.g. "RELIANCE"
        csv_sheets[f"{safe}_raw.csv"] = f"{name} Raw"

    csv_sheets["combined_raw.csv"]        = "Combined Close"
    csv_sheets["combined_normalized.csv"] = "Combined Normalized"

    print()
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for csv_file, sheet_name in csv_sheets.items():
            csv_path = os.path.join(DATA_DIR, csv_file)
            if os.path.exists(csv_path) and os.path.getsize(csv_path) > 100:
                df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                df.to_excel(writer, sheet_name=sheet_name)
                print(f"  → Sheet '{sheet_name}' written  ({len(df)} rows)")
            else:
                print(f"  ⚠ Skipped '{csv_file}' — not found or empty")

    print(f"\n  Saved → {excel_path}")
    return excel_path


# ============================================================
# Main Pipeline
# ============================================================
if __name__ == "__main__":
    import traceback
    try:
        print("=" * 55)
        print("  Stock Prediction Pipeline  (STFT + CNN)")
        print("=" * 55)

        print("\nStep 1/6 : Data Collection & Normalization")
        data, scalers = collect_data()
        export_to_excel()

        print("\nStep 2/6 : Generating STFT Spectrograms")
        spectrograms = generate_spectrograms(data)

        print("\nStep 3/6 : Visualizing")
        visualize(data, spectrograms)

        print("\nStep 4/6 : Preparing CNN Dataset")
        X_train, X_test, y_train, y_test = prepare_dataset(data, spectrograms)

        print("\nStep 4/6 : Building CNN Model")
        model = build_cnn_model(input_shape=X_train.shape[1:])

        print("\nStep 5/6 : Training")
        history = train_model(model, X_train, y_train)

        print("\nStep 6/6 : Evaluation")
        metrics = evaluate_model(model, X_test, y_test)

        print("\n" + "=" * 55)
        print("  Pipeline Complete!")
        print(f"  Final MSE  : {metrics['mse']:.6f}")
        print(f"  Final RMSE : {metrics['rmse']:.6f}")
        print(f"  Final MAE  : {metrics['mae']:.6f}")
        print("=" * 55)
        print("\n  Check the images/ folder for all plots.")
        print("  Check the models/ folder for saved model.")

    except Exception:
        print("\n!!! ERROR !!!")
        traceback.print_exc()
