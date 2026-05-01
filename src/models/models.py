"""Model definitions, training, and saving/loading logic.

Save format: each fitted model writes to its own folder under ``outputs/models/<name>/``,
always containing ``config.json`` plus algorithm-native artifacts:

  XGBoost (single output) -> model.json
  XGBoost (multi output)  -> estimator_0.json ... estimator_{n-1}.json
  MLP                     -> weights.json
  AutoencoderKNN          -> encoder.pt + train_emb.npy + train_targets.npy
  Ensemble                -> just config.json (it's a wrapper over other saved models)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def _to_array(X) -> np.ndarray:
    return X.values if hasattr(X, "values") else np.asarray(X)



# XGBoost — single output

class XGBSingle:
    type_name = "xgboost_single"

    def __init__(self) -> None:
        self.model: xgb.XGBRegressor | None = None
        self.feature_names: list[str] | None = None

    def fit(self, X, y) -> "XGBSingle":
        self.feature_names = list(X.columns) if hasattr(X, "columns") else None
        self.model = xgb.XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=4,
            subsample=0.9, colsample_bytree=0.9, n_jobs=-1, random_state=42,
            tree_method="hist",
        )
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def save(self, dir_path: Path) -> None:
        d = Path(dir_path); d.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(d / "model.json"))
        cfg = {
            "type": self.type_name,
            "n_estimators": int(self.model.n_estimators),
            "max_depth": int(self.model.max_depth),
            "learning_rate": float(self.model.learning_rate),
            "feature_names": self.feature_names,
        }
        (d / "config.json").write_text(json.dumps(cfg, indent=2))

    @classmethod
    def load(cls, dir_path: Path) -> "XGBSingle":
        d = Path(dir_path)
        cfg = json.loads((d / "config.json").read_text())
        obj = cls()
        obj.feature_names = cfg.get("feature_names")
        obj.model = xgb.XGBRegressor()
        obj.model.load_model(str(d / "model.json"))
        return obj



# XGBoost — multi output (one estimator per target stat)

class XGBMulti:
    type_name = "xgboost_multi"

    def __init__(self) -> None:
        self.estimators: list[xgb.XGBRegressor] = []
        self.feature_names: list[str] | None = None

    def fit(self, X, y) -> "XGBMulti":
        y = np.asarray(y)
        self.feature_names = list(X.columns) if hasattr(X, "columns") else None
        self.estimators = []
        for j in range(y.shape[1]):
            m = xgb.XGBRegressor(
                n_estimators=400, learning_rate=0.05, max_depth=4,
                subsample=0.9, colsample_bytree=0.9, n_jobs=-1, random_state=42,
                tree_method="hist",
            )
            m.fit(X, y[:, j])
            self.estimators.append(m)
        return self

    def predict(self, X):
        return np.column_stack([est.predict(X) for est in self.estimators])

    def save(self, dir_path: Path) -> None:
        d = Path(dir_path); d.mkdir(parents=True, exist_ok=True)
        for i, est in enumerate(self.estimators):
            est.save_model(str(d / f"estimator_{i}.json"))
        cfg = {
            "type": self.type_name,
            "n_outputs": len(self.estimators),
            "n_estimators": int(self.estimators[0].n_estimators),
            "max_depth": int(self.estimators[0].max_depth),
            "learning_rate": float(self.estimators[0].learning_rate),
            "feature_names": self.feature_names,
        }
        (d / "config.json").write_text(json.dumps(cfg, indent=2))

    @classmethod
    def load(cls, dir_path: Path) -> "XGBMulti":
        d = Path(dir_path)
        cfg = json.loads((d / "config.json").read_text())
        obj = cls()
        obj.feature_names = cfg.get("feature_names")
        for i in range(cfg["n_outputs"]):
            m = xgb.XGBRegressor()
            m.load_model(str(d / f"estimator_{i}.json"))
            obj.estimators.append(m)
        return obj



# MLP — weights stored as JSON, custom forward-prop predict so loading
# doesn't depend on sklearn's internal state machinery.

class MLPModel:
    type_name = "mlp"

    def __init__(self) -> None:
        self.scaler: StandardScaler | None = None
        self.coefs: list[np.ndarray] = []
        self.intercepts: list[np.ndarray] = []
        self.activation: str = "relu"
        self.hidden_layer_sizes: list[int] = []
        self.n_iter: int | None = None
        self.feature_names: list[str] | None = None

    def fit(self, X, y) -> "MLPModel":
        self.feature_names = list(X.columns) if hasattr(X, "columns") else None
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        mlp = MLPRegressor(
            hidden_layer_sizes=(64, 32), activation="relu",
            max_iter=600, early_stopping=True, validation_fraction=0.1, random_state=42,
        )
        mlp.fit(Xs, y)
        self.coefs = [np.asarray(c, dtype=np.float64) for c in mlp.coefs_]
        self.intercepts = [np.asarray(b, dtype=np.float64) for b in mlp.intercepts_]
        self.activation = mlp.activation
        self.hidden_layer_sizes = list(mlp.hidden_layer_sizes)
        self.n_iter = int(mlp.n_iter_)
        return self

    def _activate(self, x: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return np.maximum(x, 0)
        if self.activation == "tanh":
            return np.tanh(x)
        if self.activation == "logistic":
            return 1.0 / (1.0 + np.exp(-x))
        return x

    @staticmethod
    def _soft_bound(x: np.ndarray, lo: float = 0.0, hi: float = 100.0, beta: float = 0.5) -> np.ndarray:
        """Smoothly saturate values into [lo, hi].

        Identity for x well inside the range, asymptotes smoothly to hi (and lo) at the extremes.
        Avoids hard clipping artifacts at the bounds — a 99.5 stays 99.5, a 110 becomes ~99.99,
        and the transition between is differentiable.
        """
        inv_beta = 1.0 / beta
        upper = hi - inv_beta * np.logaddexp(0.0, beta * (hi - x))
        return lo + inv_beta * np.logaddexp(0.0, beta * (upper - lo))

    def predict(self, X):
        X_arr = _to_array(X).astype(np.float64)
        Xs = (X_arr - self.scaler.mean_) / self.scaler.scale_
        a = Xs
        for i, (W, b) in enumerate(zip(self.coefs, self.intercepts)):
            a = a @ W + b
            if i < len(self.coefs) - 1:
                a = self._activate(a)
        # Soft-saturate single-output (OVR) predictions to [0, 100].
        # Multi-output models predict raw stats with varied scales, so leave them alone.
        if a.shape[1] == 1:
            a = self._soft_bound(a, lo=0.0, hi=100.0, beta=0.5)
            return a.flatten()
        return a

    def save(self, dir_path: Path) -> None:
        d = Path(dir_path); d.mkdir(parents=True, exist_ok=True)
        weights = {
            "coefs":        [c.tolist() for c in self.coefs],
            "intercepts":   [b.tolist() for b in self.intercepts],
            "scaler_mean":  self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
        }
        (d / "weights.json").write_text(json.dumps(weights))
        cfg = {
            "type": self.type_name,
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "activation": self.activation,
            "n_iter": self.n_iter,
            "input_dim": int(self.coefs[0].shape[0]),
            "output_dim": int(self.coefs[-1].shape[1]),
            "total_parameters": int(sum(c.size + b.size for c, b in zip(self.coefs, self.intercepts))),
            "feature_names": self.feature_names,
        }
        (d / "config.json").write_text(json.dumps(cfg, indent=2))

    @classmethod
    def load(cls, dir_path: Path) -> "MLPModel":
        d = Path(dir_path)
        cfg = json.loads((d / "config.json").read_text())
        w = json.loads((d / "weights.json").read_text())
        obj = cls()
        obj.coefs = [np.array(c) for c in w["coefs"]]
        obj.intercepts = [np.array(b) for b in w["intercepts"]]
        scaler = StandardScaler()
        scaler.mean_ = np.array(w["scaler_mean"])
        scaler.scale_ = np.array(w["scaler_scale"])
        scaler.n_features_in_ = len(scaler.mean_)
        obj.scaler = scaler
        obj.activation = cfg["activation"]
        obj.hidden_layer_sizes = cfg["hidden_layer_sizes"]
        obj.n_iter = cfg.get("n_iter")
        obj.feature_names = cfg.get("feature_names")
        return obj



# Autoencoder + KNN

class _Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 8) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, input_dim),
        )

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class AutoencoderKNN:
    type_name = "autoencoder_knn"

    def __init__(self, k: int = 10, latent_dim: int = 8, epochs: int = 200, lr: float = 1e-3, batch_size: int = 128) -> None:
        self.k = k
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.scaler: StandardScaler | None = None
        self.autoencoder: _Autoencoder | None = None
        self.train_emb: np.ndarray | None = None
        self.train_targets: np.ndarray | None = None
        self.input_dim: int | None = None
        self.feature_names: list[str] | None = None

    def fit(self, X, y) -> "AutoencoderKNN":
        self.feature_names = list(X.columns) if hasattr(X, "columns") else None
        X_arr = _to_array(X)
        self.input_dim = X_arr.shape[1]

        self.scaler = StandardScaler().fit(X_arr)
        Xs = self.scaler.transform(X_arr).astype("float32")
        Xt = torch.tensor(Xs)

        torch.manual_seed(42)
        self.autoencoder = _Autoencoder(input_dim=self.input_dim, latent_dim=self.latent_dim)
        opt = torch.optim.Adam(self.autoencoder.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        ds = torch.utils.data.TensorDataset(Xt)
        dl = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        self.autoencoder.train()
        for _ in range(self.epochs):
            for (batch,) in dl:
                opt.zero_grad()
                loss = loss_fn(self.autoencoder(batch), batch)
                loss.backward()
                opt.step()

        self.autoencoder.eval()
        with torch.no_grad():
            self.train_emb = self.autoencoder.encode(Xt).numpy()
        self.train_targets = np.asarray(y)
        return self

    def predict(self, X):
        X_arr = _to_array(X)
        Xs = self.scaler.transform(X_arr).astype("float32")
        Xt = torch.tensor(Xs)
        with torch.no_grad():
            emb = self.autoencoder.encode(Xt).numpy()
        nn_idx = NearestNeighbors(n_neighbors=self.k).fit(self.train_emb)
        _, idxs = nn_idx.kneighbors(emb)
        return self.train_targets[idxs].mean(axis=1)

    def save(self, dir_path: Path) -> None:
        d = Path(dir_path); d.mkdir(parents=True, exist_ok=True)
        torch.save(self.autoencoder.state_dict(), d / "encoder.pt")
        np.save(d / "train_emb.npy", self.train_emb)
        np.save(d / "train_targets.npy", self.train_targets)
        cfg = {
            "type": self.type_name,
            "k": self.k,
            "latent_dim": self.latent_dim,
            "epochs": self.epochs,
            "input_dim": self.input_dim,
            "scaler_mean":  self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
            "n_train": int(self.train_emb.shape[0]),
            "target_shape": list(self.train_targets.shape),
            "feature_names": self.feature_names,
        }
        (d / "config.json").write_text(json.dumps(cfg, indent=2))

    @classmethod
    def load(cls, dir_path: Path) -> "AutoencoderKNN":
        d = Path(dir_path)
        cfg = json.loads((d / "config.json").read_text())
        obj = cls(k=cfg["k"], latent_dim=cfg["latent_dim"], epochs=cfg.get("epochs", 200))
        obj.input_dim = cfg["input_dim"]
        obj.feature_names = cfg.get("feature_names")
        scaler = StandardScaler()
        scaler.mean_ = np.array(cfg["scaler_mean"])
        scaler.scale_ = np.array(cfg["scaler_scale"])
        scaler.n_features_in_ = len(scaler.mean_)
        obj.scaler = scaler
        obj.autoencoder = _Autoencoder(input_dim=obj.input_dim, latent_dim=obj.latent_dim)
        obj.autoencoder.load_state_dict(torch.load(d / "encoder.pt", weights_only=True))
        obj.autoencoder.eval()
        obj.train_emb = np.load(d / "train_emb.npy")
        obj.train_targets = np.load(d / "train_targets.npy")
        return obj



# Ensemble

class EnsembleAverage:
    type_name = "ensemble_avg"

    def __init__(self, models: list | None = None, model_names: list[str] | None = None) -> None:
        self.models = models or []
        self.model_names = model_names or []

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models], axis=0)

    def save(self, dir_path: Path) -> None:
        d = Path(dir_path); d.mkdir(parents=True, exist_ok=True)
        cfg = {"type": self.type_name, "components": self.model_names}
        (d / "config.json").write_text(json.dumps(cfg, indent=2))

    @classmethod
    def load(cls, dir_path: Path, components_resolver) -> "EnsembleAverage":
        d = Path(dir_path)
        cfg = json.loads((d / "config.json").read_text())
        models = [components_resolver(n) for n in cfg["components"]]
        return cls(models=models, model_names=cfg["components"])


# Public training API + uniform loader

def train_xgb(X, y):
    """Auto-dispatch: XGBSingle for 1d y, XGBMulti for 2d y."""
    if np.asarray(y).ndim == 1:
        return XGBSingle().fit(X, y)
    return XGBMulti().fit(X, y)


def train_mlp(X, y) -> MLPModel:
    return MLPModel().fit(X, y)


def train_autoencoder_knn(X, y, k: int = 10, epochs: int = 200) -> AutoencoderKNN:
    return AutoencoderKNN(k=k, epochs=epochs).fit(X, y)


def train_ensemble(*models, names: list[str] | None = None) -> EnsembleAverage:
    if names is None:
        names = [getattr(m, "type_name", type(m).__name__) for m in models]
    return EnsembleAverage(models=list(models), model_names=names)


_KIND_TO_LOADER = {
    XGBSingle.type_name:      lambda d, _r: XGBSingle.load(d),
    XGBMulti.type_name:       lambda d, _r: XGBMulti.load(d),
    MLPModel.type_name:       lambda d, _r: MLPModel.load(d),
    AutoencoderKNN.type_name: lambda d, _r: AutoencoderKNN.load(d),
    EnsembleAverage.type_name: lambda d, r: EnsembleAverage.load(d, r),
}


def load_model(name: str, models_dir: Path):
    """Load a saved model by directory name. Recurses into ensemble components."""
    d = Path(models_dir) / name
    if not d.is_dir():
        raise FileNotFoundError(f"No model directory at {d}")
    cfg = json.loads((d / "config.json").read_text())
    kind = cfg["type"]
    if kind not in _KIND_TO_LOADER:
        raise ValueError(f"Unknown model type: {kind}")
    return _KIND_TO_LOADER[kind](d, lambda n: load_model(n, models_dir))
