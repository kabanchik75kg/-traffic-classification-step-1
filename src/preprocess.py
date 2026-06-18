# src/preprocess.py
# Препроцессинг-слой (наша доп-разметка). Два представления признаков:
#
#   tree-view   (RF, XGBoost):  сырьё как есть — sentinel -1 сохранён,
#                               ip_prot числом, без масштабирования.
#                               Это РОВНО то, что отдаёт iter_batches
#                               (76 колонок),
#                               отдельный объект не нужен.
#
#   linear-view (LogReg, MLP):  StandardScaler на непрерывных
#                               + one-hot ip_prot (4 категории {1,2,6,17})
#                               + indicator bwd_tcp_init_win_absent
#                               + замена sentinel -1 -> 0 в TCP-window колонках
#                               Итог: 76 - 1(ip_prot) + 4(one-hot) +
#                               1(indicator) = 80 колонок.
#
# КЛЮЧЕВОЕ: препроцессор фитится СТРОГО на train и сохраняется через joblib.
# Cross-dataset (LycoS17) обязан пройти ровно те же трансформации с теми же
# параметрами (mean/std скейлера, категории one-hot), иначе результат невалиден

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import joblib  # type: ignore
from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # type: ignore

from src import config


# Явные категории протокола — гарантируют 4 столбца независимо
# от выборки для fit.
# handle_unknown='ignore' → протокол, отсутствующий в train
# (напр. SCTP в LycoS17),
# кодируется нулевым вектором, а не ломает размерность.
PROTO_CATEGORIES = [1.0, 2.0, 6.0, 17.0]   # ICMP, IGMP, TCP, UDP


class FlowPreprocessor(BaseEstimator, TransformerMixin):
    """
    linear/nn-представление flow-признаков.

    На вход — матрица в порядке feature_cols (как отдаёт iter_batches).
    На выход — [scaled_continuous | onehot(ip_prot) | indicator] во float32.

    Порядок операций при transform важен:
      1. indicator считается ДО замены sentinel (иначе сигнал теряется);
      2. -1 -> 0 в TCP-window ДО скейлинга (иначе -1 искажает mean/std);
      3. скейлинг непрерывных + one-hot протокола.
    """

    def __init__(self, feature_cols,
                 proto_col=config.PROTO_COL,
                 tcp_win_cols=tuple(config.TCP_WIN_COLS),
                 indicator_src_col=config.TCP_WIN_INDICATOR_COL,
                 sentinel=config.SENTINEL_VALUE,
                 proto_categories=tuple(PROTO_CATEGORIES)):
        self.feature_cols = list(feature_cols)
        self.proto_col = proto_col
        self.tcp_win_cols = list(tcp_win_cols)
        self.indicator_src_col = indicator_src_col
        self.sentinel = sentinel
        self.proto_categories = list(proto_categories)

        # Непрерывные = все признаки, кроме категориального ip_prot.
        # TCP-window колонки сюда входят (после -1->0 они непрерывные).
        self.continuous_cols = [
            c for c in self.feature_cols if c != self.proto_col
        ]

    # ── вспомогательное ──
    def _to_df(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.feature_cols]
        return pd.DataFrame(X, columns=self.feature_cols)

    def _clean_sentinel(self, df):
        """Замена sentinel -1 -> 0 в TCP-window колонках (на копии)."""
        df = df.copy()
        for c in self.tcp_win_cols:
            col = df[c].to_numpy()
            df[c] = np.where(col == self.sentinel, 0.0, col)
        return df

    # ── fit ──
    def fit(self, X, y=None):
        df = self._to_df(X)
        df_clean = self._clean_sentinel(df)

        self.scaler_ = StandardScaler().fit(
            df_clean[self.continuous_cols].to_numpy(dtype=np.float64)
        )
        self.encoder_ = OneHotEncoder(
            categories=[self.proto_categories],
            handle_unknown="ignore",
            sparse_output=False,
            dtype=np.float32,
        ).fit(df[[self.proto_col]].to_numpy())

        self.n_features_out_ = (
            len(self.continuous_cols) + len(self.proto_categories) + 1
        )
        return self

    # ── transform ──
    def transform(self, X):
        df = self._to_df(X)

        # 1. indicator — ДО очистки sentinel.
        indicator = (
            df[self.indicator_src_col].to_numpy() == self.sentinel
        ).astype(np.float32).reshape(-1, 1)

        # 2. -1 -> 0, затем скейлинг непрерывных.
        df_clean = self._clean_sentinel(df)
        cont = self.scaler_.transform(
            df_clean[self.continuous_cols].to_numpy(dtype=np.float64)
        ).astype(np.float32)

        # 3. one-hot протокола.
        onehot = self.encoder_.transform(df[[self.proto_col]].to_numpy())

        return np.hstack([cont, onehot, indicator]).astype(np.float32)

    # ── имена выходных колонок (для интерпретации весов LogReg) ──
    def get_feature_names_out(self, input_features=None):
        proto_names = [f"{self.proto_col}_{int(c)}"
                       for c in self.proto_categories]
        return np.array(
            self.continuous_cols + proto_names + ["bwd_tcp_init_win_absent"]
        )


# ── фит на подвыборке train + сохранение ──
def fit_preprocessor(con, feature_cols, *,
                     train_path=None,
                     sample_size=2_000_000,
                     save_path=None):
    """
    Фитит FlowPreprocessor на подвыборке train и (опц.) сохраняет через joblib.

    Файл сплитов уже детерминированно перемешан, поэтому первые
    sample_size строк — это случайная репрезентативная выборка.
    2M строк более чем достаточно для
    устойчивых mean/std и присутствия всех 4 протоколов.
    """
    train_path = train_path or config.TRAIN_PATH
    save_path = save_path or (config.MODELS_DIR / "preprocessor.pkl")

    cols_sql = ", ".join('"' + c.replace('"', '""') + '"'
                         for c in feature_cols)
    df = con.execute(
        f"SELECT {cols_sql}"
        f"FROM read_parquet('{train_path}')"
        f"LIMIT {sample_size}"
    ).df()

    pre = FlowPreprocessor(feature_cols).fit(df)

    if save_path is not None:
        joblib.dump(pre, save_path)

    return pre, save_path


def load_preprocessor(path=None):
    path = path or (config.MODELS_DIR / "preprocessor.pkl")
    return joblib.load(path)
