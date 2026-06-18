# src/config.py
# Единая точка правды: пути, имена колонок, константы.
# Меняется здесь — подхватывается всеми ноутбуками и модулями.

from pathlib import Path

# ── Пути (относительно корня проекта; ноутбуки запускаются из notebooks/)
# В ноутбуке: sys.path.insert(0, ".."), поэтому базой считаем родителя src/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = PROCESSED_DIR / "splits"

MODELS_DIR = PROJECT_ROOT / "models"

RESULTS_DIR = PROJECT_ROOT / "results"
BASELINE_DIR = RESULTS_DIR / "baseline"
CROSS_DIR = RESULTS_DIR / "cross_dataset"

TRAIN_PATH = (SPLITS_DIR / "train.parquet").as_posix()
VAL_PATH = (SPLITS_DIR / "val.parquet").as_posix()
TEST_PATH = (SPLITS_DIR / "test.parquet").as_posix()

# Создаём выходные директории при импорте (data/* уже существуют).
for _d in (MODELS_DIR, BASELINE_DIR, CROSS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Имена целевых колонок ─────────────────
# Канон задан ноутбуком 1: label (исходная) + label_binary (бинарная цель).
LABEL_COL = "label"
BINARY_LABEL_COL = "label_binary"
TARGET_COLS = (LABEL_COL, BINARY_LABEL_COL)
BENIGN_LABEL = "Benign"

# ── Ожидаемое число признаков (защита от рассинхрона схемы) ──────────────
# processed = 78 колонок: 76 признаков + label + label_binary.
EXPECTED_N_FEATURES = 76

# ── Признаки, требующие особой обработки в linear/nn-препроцессинге ───────
PROTO_COL = "ip_prot"                  # one-hot для linear/nn
TCP_WIN_COLS = ["fwd_tcp_init_win_bytes", "bwd_tcp_init_win_bytes"]
TCP_WIN_INDICATOR_COL = "bwd_tcp_init_win_bytes"
# из него делаем indicator "окно отсутствует"
SENTINEL_VALUE = -1                          # маркер "значение отсутствует"

# ── Прочее ────────────────────────────────────────────────────────
BATCH_SIZE = 500_000
RANDOM_STATE = 42
DUCKDB_MEMORY = "4GB"
DUCKDB_TEMP = "/tmp/duckdb_spill"

LABEL_MAP_2017 = {
    "benign":                  "Benign",
    "ftp_patator":             "FTP-Patator",
    "ssh_patator":             "SSH-Patator",
    "dos_hulk":                "DoS Hulk",
    "dos_goldeneye":           "DoS GoldenEye",
    "dos_slowloris":           "DoS Slowloris",
    "dos_slowhttptest":        "DoS Slowhttptest",
    "webattack_bruteforce":    "Web Attack - Brute Force",
    "webattack_xss":           "Web Attack - XSS",
    "webattack_sql_injection": "Web Attack - Sql Injection",
    "bot":                     "Bot",
    "ddos":                    "DDoS",  # 2017: общий DDoS (в 2018 три подтипа)
    "heartbleed":              "Heartbleed",    # только 2017
    "portscan":                "PortScan",      # только 2017
}

CROSS_2017_PATH = (PROCESSED_DIR / "2017.parquet").as_posix()
RAW_2017_DIR = (RAW_DIR / "lycos-ids2017").as_posix()
