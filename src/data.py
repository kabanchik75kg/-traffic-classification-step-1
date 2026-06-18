# src/data.py
# Доступ к данным: DuckDB-коннект, список признаков с проверкой инвариантов,
# потоковый итератор батчей. Используется всеми обучающими ноутбуками.

import gc

import duckdb
import numpy as np
import pyarrow.parquet as pq

from src import config


def get_duckdb_connection():
    """DuckDB-коннект с настройками памяти и spill-директории из config."""
    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{config.DUCKDB_MEMORY}'")
    con.execute(f"SET temp_directory = '{config.DUCKDB_TEMP}'")
    return con


def get_feature_cols(con, parquet_path=None, *, strict=True):
    """
    Возвращает список признаков = все колонки, КРОМE label и label_binary.

    Защита от двух ошибок, которые иначе проходят молча:
      1. таргет (label / label_binary) просачивается в признаки → утечка,
         идеальные метрики, невалидная модель;
      2. схема разъехалась (не 76 признаков) → рассинхрон с processed.

    strict=True (по умолчанию) — несоответствие поднимает AssertionError.
    strict=False — только печатает предупреждение (для разведки/отладки).
    """
    parquet_path = parquet_path or config.TRAIN_PATH

    all_cols = con.execute(
        f"SELECT * FROM read_parquet('{parquet_path}') LIMIT 0"
    ).df().columns.tolist()

    feature_cols = [c for c in all_cols if c not in config.TARGET_COLS]

    leaked = set(feature_cols) & set(config.TARGET_COLS)
    n_ok = (len(feature_cols) == config.EXPECTED_N_FEATURES)

    if strict:
        assert not leaked, (
            f"Утечка таргета в признаки: {leaked}. "
            f"Проверь имена целевых колонок (ожидаются {config.TARGET_COLS})."
        )
        assert n_ok, (
            f"Ожидалось {config.EXPECTED_N_FEATURES} признаков, "
            f"получено {len(feature_cols)}. Схема разъехалась с processed."
        )
    else:
        if leaked:
            print(f"[warn] таргет среди признаков: {leaked}")
        if not n_ok:
            print(f"[warn] признаков {len(feature_cols)}, "
                  f"ожидали {config.EXPECTED_N_FEATURES}")

    return feature_cols


def count_rows(con, parquet_path):
    """Число строк в parquet (по метаданным, без полного чтения)."""
    return con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{parquet_path}')"
    ).fetchone()[0]


def iter_batches(path, feature_cols, *, batch_size=None, with_label=False):
    """
    Потоковый итератор батчей через PyArrow (файл остаётся на диске).

    Отдаёт:
      with_label=False → (X float32, y int8)
      with_label=True  → (X float32, y int8, lbl object)
      — lbl для разреза по классам

    X — только feature_cols в заданном порядке (порядок фиксирован вызывающим
    кодом, что важно для согласованности с обученным препроцессором/моделью).
    """
    batch_size = batch_size or config.BATCH_SIZE
    req_cols = list(feature_cols) + [config.BINARY_LABEL_COL]
    if with_label:
        req_cols.append(config.LABEL_COL)

    pf = pq.ParquetFile(path)
    for rb in pf.iter_batches(batch_size=batch_size, columns=req_cols):
        df = rb.to_pandas()
        X = df[feature_cols].values.astype(np.float32)
        y = df[config.BINARY_LABEL_COL].values.astype(np.int8)
        if with_label:
            lbl = df[config.LABEL_COL].values
            del df, rb
            gc.collect()
            yield X, y, lbl
        else:
            del df, rb
            gc.collect()
            yield X, y
