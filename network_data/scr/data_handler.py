"""
Network intrusion detection 데이터셋 통합 로더.

8개 데이터셋을 동일 인터페이스로 로드:
    load_network_dataset(name) -> (X_tr, X_te, y_tr, y_te, class_counts, num_classes, class_names)

- generic 6종 (Bot-IoT, TON_IoT, RT-IoT2022, CICIDS2017/2018, CIC-DDoS2019):
  kagglehub 다운로드 → 라벨 자동감지 → 숫자피처 → stratified subset(300K, BENIGN 10만 상한)
  → 70/30 stratified split → StandardScaler
- NSL-KDD / UNSW-NB15: 공식 train/test split + 범주형 one-hot

tabular_data/scr/data_handler.py 와 동일한 (X_tr, X_te, y_tr, y_te) 인터페이스를 따른다.
각 노트북에 흩어져 있던 검증된 전처리를 그대로 포팅한 것이며, 동작은 기존 8개 노트북과 동일.
"""

import os
import glob
import gc
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

TARGET_TOTAL = 300_000   # 대용량 데이터셋 subset 크기
BENIGN_MAX   = 100_000   # 정상/BENIGN 클래스 최대 샘플 수
SEED         = 42
TEST_SIZE    = 0.30      # train/test split (tabular와 동일)

DATASETS = ['NSL-KDD', 'UNSW-NB15', 'CICIDS2017', 'CICIDS2018',
            'CIC-DDoS2019', 'Bot-IoT', 'TON_IoT', 'RT-IoT2022']


# ==================================================================
# 공유 헬퍼
# ==================================================================
def _detect_files(path):
    all_files = sorted(f for f in glob.glob(f'{path}/**/*', recursive=True)
                       if os.path.isfile(f))
    print(f'All files found ({len(all_files)}):')
    for f in all_files[:20]:
        print(f'  {os.path.relpath(f, path)}  ({os.path.getsize(f):,} bytes)')
    if len(all_files) > 20:
        print(f'  ... and {len(all_files) - 20} more')
    csv_files     = sorted(glob.glob(f'{path}/**/*.csv',     recursive=True))
    parquet_files = sorted(glob.glob(f'{path}/**/*.parquet', recursive=True))
    return all_files, csv_files, parquet_files


def _detect_label_col(df, priority):
    for c in priority:
        if c in df.columns:
            return c
    col = df.columns[-1]
    print(f'Warning: using last column as label: {col!r}')
    return col


def _stratified_subset(df, label_col, target_total, benign_max):
    """정상 클래스는 benign_max로 상한, 나머지는 비율 유지(최소 50개)로 추출."""
    unique_labels = df[label_col].unique()
    normal_label  = next((l for l in unique_labels
                          if 'normal' in str(l).lower() or 'benign' in str(l).lower()),
                         None)
    non_budget = target_total - (benign_max if normal_label is not None else 0)
    non_total  = (df[label_col] != normal_label).sum() if normal_label is not None else len(df)
    rng = np.random.RandomState(SEED)
    idx_list = []
    for lbl in unique_labels:
        idx = df.index[df[label_col] == lbl].tolist()
        if normal_label is not None and lbl == normal_label:
            take = min(len(idx), benign_max)
        else:
            prop = len(idx) / max(non_total, 1)
            take = max(50, min(len(idx), int(prop * non_budget)))
        take = min(take, len(idx))
        idx_list.extend(rng.choice(idx, take, replace=False).tolist())
    return df.loc[idx_list].reset_index(drop=True)


def _read_csv_chunked(f, rows_per_file):
    """파일 전체에 걸쳐 균등 샘플링 (공격 트래픽이 파일 후반부에 몰린 경우 대응)."""
    sample_per_chunk = max(500, rows_per_file // 6)
    for enc in ('utf-8', 'latin-1'):
        try:
            chunks = [chunk.sample(min(sample_per_chunk, len(chunk)), random_state=SEED)
                      for chunk in pd.read_csv(f, chunksize=100_000, encoding=enc, low_memory=False)]
            break
        except UnicodeDecodeError:
            continue
    df_part = pd.concat(chunks, ignore_index=True)
    if len(df_part) > rows_per_file:
        df_part = df_part.sample(rows_per_file, random_state=SEED)
    return df_part


def _read_csv(f, nrows=None):
    for enc in ('utf-8', 'latin-1'):
        try:
            return pd.read_csv(f, encoding=enc, low_memory=False, nrows=nrows)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f'cannot decode {f}')


def _summarize(class_names, class_counts, X_tr, X_te):
    print(f'\n✓ Loaded: Train={len(X_tr):,}, Test={len(X_te):,}, Input dim={X_tr.shape[1]}')
    print('Class distribution (train):')
    total = sum(class_counts)
    for i, (name, cnt) in enumerate(zip(class_names, class_counts)):
        print(f'  {i:2d} {str(name):35s}: {cnt:7,d} ({cnt / total * 100:.2f}%)')
    nz = [c for c in class_counts if c > 0]
    ir = max(class_counts) / max(min(nz), 1) if nz else 0
    print(f'Imbalance ratio: {ir:.0f}:1')


def _finalize(X, y, class_names):
    """단일 행렬 → 70/30 stratified split + StandardScaler + class_counts."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr).astype(np.float32)
    X_te = scaler.transform(X_te).astype(np.float32)
    num_cls = len(class_names)
    class_counts = [int((y_tr == i).sum()) for i in range(num_cls)]
    _summarize(class_names, class_counts, X_tr, X_te)
    return X_tr, X_te, y_tr.astype(np.int64), y_te.astype(np.int64), class_counts, num_cls, class_names


def _finalize_predef(X_tr, X_te, y_tr, y_te, class_names):
    """공식 train/test split 사용 데이터셋(NSL-KDD, UNSW)용: scale + class_counts."""
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr).astype(np.float32)
    X_te = scaler.transform(X_te).astype(np.float32)
    num_cls = len(class_names)
    class_counts = [int((y_tr == i).sum()) for i in range(num_cls)]
    _summarize(class_names, class_counts, X_tr, X_te)
    return X_tr, X_te, y_tr.astype(np.int64), y_te.astype(np.int64), class_counts, num_cls, class_names


# ==================================================================
# NSL-KDD (공식 train/test + 범주형 one-hot)
# ==================================================================
_KDD_COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty',
]
_KDD_CLASS_NAMES = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']
_DOS   = {'back', 'land', 'neptune', 'pod', 'smurf', 'teardrop',
          'apache2', 'udpstorm', 'processtable', 'worm'}
_PROBE = {'satan', 'ipsweep', 'nmap', 'portsweep', 'mscan', 'saint'}
_R2L   = {'guess_passwd', 'ftp_write', 'imap', 'phf', 'multihop',
          'warezmaster', 'warezclient', 'spy', 'xlock', 'xsnoop',
          'snmpguess', 'snmpgetattack', 'httptunnel', 'sendmail', 'named'}
_U2R   = {'buffer_overflow', 'loadmodule', 'perl', 'rootkit',
          'xterm', 'ps', 'sqlattack'}


def _kdd_map(lbl):
    lbl = str(lbl).strip().lower()
    if lbl == 'normal':   return 0
    if lbl in _DOS:       return 1
    if lbl in _PROBE:     return 2
    if lbl in _R2L:       return 3
    if lbl in _U2R:       return 4
    return 1  # 미분류 공격 → DoS


def load_nslkdd():
    import kagglehub
    print('Downloading NSL-KDD...')
    path = kagglehub.dataset_download('hassan06/nslkdd')
    files = glob.glob(f'{path}/**/*', recursive=True)
    train_file = next(f for f in files if os.path.basename(f) == 'KDDTrain+.txt')
    test_file  = next(f for f in files if os.path.basename(f) == 'KDDTest+.txt')

    df_train = pd.read_csv(train_file, names=_KDD_COLUMNS, header=None)
    df_test  = pd.read_csv(test_file,  names=_KDD_COLUMNS, header=None)
    df_train['class'] = df_train['label'].apply(_kdd_map)
    df_test['class']  = df_test['label'].apply(_kdd_map)

    # 범주형 3개 → one-hot (train+test 합쳐서 인코딩, 컬럼 일관성 보장)
    cat_cols = ['protocol_type', 'service', 'flag']
    df_all = pd.concat([df_train, df_test], axis=0, ignore_index=True)
    df_all = pd.get_dummies(df_all, columns=cat_cols, dtype=np.float32)
    n_train = len(df_train)
    df_tr_enc, df_te_enc = df_all.iloc[:n_train], df_all.iloc[n_train:]

    feat = [c for c in df_tr_enc.columns if c not in ('label', 'difficulty', 'class')]
    X_tr = df_tr_enc[feat].values.astype(np.float32)
    y_tr = df_tr_enc['class'].values.astype(np.int64)
    X_te = df_te_enc[feat].values.astype(np.float32)
    y_te = df_te_enc['class'].values.astype(np.int64)
    return _finalize_predef(X_tr, X_te, y_tr, y_te, _KDD_CLASS_NAMES)


# ==================================================================
# UNSW-NB15 (공식 train/test + 범주형 one-hot)
# ==================================================================
def load_unsw():
    import kagglehub
    print('Downloading UNSW-NB15...')
    path = kagglehub.dataset_download('mrwellsdavid/unsw-nb15')
    train_files = glob.glob(f'{path}/**/UNSW_NB15_training-set.csv', recursive=True)
    test_files  = glob.glob(f'{path}/**/UNSW_NB15_testing-set.csv',  recursive=True)
    if not train_files or not test_files:
        all_csv = sorted(glob.glob(f'{path}/**/*.csv', recursive=True))
        train_files = [f for f in all_csv if 'training' in f.lower()]
        test_files  = [f for f in all_csv if 'testing'  in f.lower()]

    df_train = pd.read_csv(train_files[0], low_memory=False)
    df_test  = pd.read_csv(test_files[0],  low_memory=False)

    for df in (df_train, df_test):
        df['attack_cat'] = df['attack_cat'].astype(str).str.strip()
        df.loc[df['attack_cat'].isin(['', 'nan', 'NaN']), 'attack_cat'] = 'Normal'

    cat_cols  = ['proto', 'service', 'state']
    drop_cols = ['id', 'label']
    df_all = pd.concat([df_train, df_test], axis=0, ignore_index=True)
    df_all = pd.get_dummies(df_all, columns=cat_cols, dtype=np.float32)
    n_train = len(df_train)
    df_tr_enc = df_all.iloc[:n_train].copy()
    df_te_enc = df_all.iloc[n_train:].copy()

    feat = [c for c in df_tr_enc.columns if c not in drop_cols + ['attack_cat']]
    for df in (df_tr_enc, df_te_enc):
        df[feat] = df[feat].replace([np.inf, -np.inf], np.nan)
    df_tr_enc = df_tr_enc.dropna(subset=feat).reset_index(drop=True)
    df_te_enc = df_te_enc.dropna(subset=feat).reset_index(drop=True)
    num_cols = df_tr_enc[feat].select_dtypes(include=[np.number]).columns.tolist()

    le = LabelEncoder()
    le.fit(pd.concat([df_tr_enc['attack_cat'], df_te_enc['attack_cat']]))
    y_tr = le.transform(df_tr_enc['attack_cat']).astype(np.int64)
    y_te = le.transform(df_te_enc['attack_cat']).astype(np.int64)
    X_tr = df_tr_enc[num_cols].values.astype(np.float32)
    X_te = df_te_enc[num_cols].values.astype(np.float32)
    return _finalize_predef(X_tr, X_te, y_tr, y_te, le.classes_.tolist())


# ==================================================================
# CICIDS2017 (요일별 CSV, BENIGN 상한 subset)
# ==================================================================
def load_cicids2017():
    import kagglehub
    print('Downloading CICIDS2017...')
    path = kagglehub.dataset_download('chethuhn/network-intrusion-dataset')
    csv_files = sorted(glob.glob(f'{path}/**/*.csv', recursive=True)) or \
                sorted(glob.glob(f'{path}/*.csv'))
    print(f'CSV files: {len(csv_files)}')
    df = pd.concat([_read_csv(f) for f in csv_files], axis=0, ignore_index=True)
    print(f'Total rows: {len(df):,}')

    label_col = [c for c in df.columns if 'label' in c.lower()][0]
    df['label'] = df[label_col].astype(str).str.strip()
    feat = [c for c in df.columns if c != label_col]
    df[feat] = df[feat].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feat)
    num_cols = df[feat].select_dtypes(include=[np.number]).columns.tolist()

    df_sub = _stratified_subset(df, 'label', TARGET_TOTAL, BENIGN_MAX)
    del df; gc.collect()
    le = LabelEncoder()
    y = le.fit_transform(df_sub['label']).astype(np.int64)
    X = df_sub[num_cols].values.astype(np.float32)
    return _finalize(X, y, le.classes_.tolist())


# ==================================================================
# CICIDS2018 (대용량 다중 CSV, 헤더 오염/inf 컬럼 robust 처리)
# ==================================================================
def load_cicids2018():
    import kagglehub
    print('Downloading CICIDS2018...')
    path = kagglehub.dataset_download('solarmainframe/ids-intrusion-csv')
    all_files, csv_files, _ = _detect_files(path)
    if not csv_files:
        raise FileNotFoundError(f'No CSV files: {[os.path.basename(f) for f in all_files]}')

    rows_per_file = max(5_000, (TARGET_TOTAL * 2) // len(csv_files))
    print(f'Loading {len(csv_files)} CSV files (~{rows_per_file:,} rows each)...')
    df = pd.concat([_read_csv_chunked(f, rows_per_file) for f in csv_files],
                   axis=0, ignore_index=True)
    print(f'Total rows loaded: {len(df):,}')

    label_col = _detect_label_col(
        df, ['Label', 'label', 'Class', 'class', 'Attack_type', 'attack_type', 'category'])
    df['label'] = df[label_col].astype(str).str.strip()
    df = df[df['label'] != label_col].reset_index(drop=True)  # 헤더-as-data 제거

    feat = [c for c in df.columns if c not in (label_col, 'label')]
    df[feat] = df[feat].apply(pd.to_numeric, errors='coerce')
    num_cols = df[feat].select_dtypes(include=[np.number]).columns.tolist()
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    inf_fracs = df[num_cols].isna().mean()
    drop_cols = inf_fracs[inf_fracs > 0.30].index.tolist()
    if drop_cols:
        print(f'Dropping {len(drop_cols)} mostly-inf columns')
        num_cols = [c for c in num_cols if c not in drop_cols]
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    df_sub = _stratified_subset(df, 'label', TARGET_TOTAL, BENIGN_MAX)
    del df; gc.collect()
    le = LabelEncoder()
    y = le.fit_transform(df_sub['label']).astype(np.int64)
    X = df_sub[num_cols].values.astype(np.float32)
    return _finalize(X, y, le.classes_.tolist())


# ==================================================================
# CIC-DDoS2019 (CSV 또는 Parquet, BENIGN 상한 subset)
# ==================================================================
def load_cicddos2019():
    import kagglehub
    print('Downloading CIC-DDoS2019...')
    path = kagglehub.dataset_download('dhoogla/cicddos2019')
    _, csv_files, parquet_files = _detect_files(path)
    if csv_files:
        df = pd.concat([_read_csv(f) for f in csv_files], axis=0, ignore_index=True)
    elif parquet_files:
        df = pd.concat([pd.read_parquet(f) for f in parquet_files], axis=0, ignore_index=True)
    else:
        raise FileNotFoundError('No CSV or Parquet files found for CIC-DDoS2019.')
    print(f'Total rows: {len(df):,}')

    label_col = [c for c in df.columns if 'label' in c.lower()][0]
    df['label'] = df[label_col].astype(str).str.strip()
    feat = [c for c in df.columns if c != label_col]
    df[feat] = df[feat].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feat)
    num_cols = df[feat].select_dtypes(include=[np.number]).columns.tolist()

    df_sub = _stratified_subset(df, 'label', TARGET_TOTAL, BENIGN_MAX)
    del df; gc.collect()
    le = LabelEncoder()
    y = le.fit_transform(df_sub['label']).astype(np.int64)
    X = df_sub[num_cols].values.astype(np.float32)
    return _finalize(X, y, le.classes_.tolist())


# ==================================================================
# Bot-IoT (대용량 다중 CSV, 'category' 라벨, feature 정의파일 제외)
# ==================================================================
def load_botiot():
    import kagglehub
    print('Downloading Bot-IoT...')
    path = kagglehub.dataset_download('vigneshvenkateswaran/bot-iot')
    all_files, csv_files, _ = _detect_files(path)
    files = [f for f in csv_files if 'feature' not in os.path.basename(f).lower()] or csv_files
    if not files:
        raise FileNotFoundError(f'No CSV files: {[os.path.basename(f) for f in all_files]}')

    rows_per_file = max(5_000, (TARGET_TOTAL * 2) // len(files))
    print(f'Loading {len(files)} CSV files, up to {rows_per_file:,} rows each...')
    df = pd.concat([_read_csv(f, nrows=rows_per_file) for f in files],
                   axis=0, ignore_index=True)
    print(f'Total rows loaded: {len(df):,}')

    label_col = _detect_label_col(
        df, ['category', 'Category', 'sub_category', 'type', 'Type',
             'Attack_type', 'attack_type', 'Label', 'label', 'attack', 'Attack'])
    df['label'] = df[label_col].astype(str).str.strip()
    if set(df['label'].unique()) <= {'0', '1', '0.0', '1.0'}:
        print('Warning: label column appears binary.')

    feat = [c for c in df.columns if c not in (label_col, 'label')]
    num_cols = df[feat].select_dtypes(include=[np.number]).columns.tolist()
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    df_sub = _stratified_subset(df, 'label', TARGET_TOTAL, BENIGN_MAX)
    del df; gc.collect()
    le = LabelEncoder()
    y = le.fit_transform(df_sub['label']).astype(np.int64)
    X = df_sub[num_cols].values.astype(np.float32)
    return _finalize(X, y, le.classes_.tolist())


# ==================================================================
# TON_IoT ('type' 라벨 소문자 정규화, 크기 작으면 subset 생략)
# ==================================================================
def load_toniot():
    import kagglehub
    print('Downloading TON_IoT...')
    path = kagglehub.dataset_download('arnobbhowmik/ton-iot-network-dataset')
    csv_files = sorted(glob.glob(f'{path}/**/*.csv', recursive=True))
    if not csv_files:
        raise FileNotFoundError(f'No CSV files in {path}')
    df = pd.concat([_read_csv(f) for f in csv_files], axis=0, ignore_index=True)
    print(f'Total rows: {len(df):,}')

    label_col = _detect_label_col(
        df, ['type', 'Type', 'attack_cat', 'Attack_type', 'attack_type',
             'Label', 'label', 'category', 'attack', 'Attack'])
    df['label'] = df[label_col].astype(str).str.strip().str.lower()
    feat = [c for c in df.columns if c not in (label_col, 'label')]
    df[feat] = df[feat].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feat)
    num_cols = df[feat].select_dtypes(include=[np.number]).columns.tolist()

    df_sub = _stratified_subset(df, 'label', TARGET_TOTAL, BENIGN_MAX) \
        if len(df) > TARGET_TOTAL else df.copy()
    del df; gc.collect()
    le = LabelEncoder()
    y = le.fit_transform(df_sub['label']).astype(np.int64)
    X = df_sub[num_cols].values.astype(np.float32)
    return _finalize(X, y, le.classes_.tolist())


# ==================================================================
# RT-IoT2022 ('Attack_type' 라벨, 크기 작으면 subset 생략)
# ==================================================================
def load_rtiot2022():
    import kagglehub
    print('Downloading RT-IoT2022...')
    path = kagglehub.dataset_download('supplejade/rt-iot2022real-time-internet-of-things')
    csv_files = sorted(glob.glob(f'{path}/**/*.csv', recursive=True))
    if not csv_files:
        raise FileNotFoundError(f'No CSV files in {path}')
    df = pd.concat([_read_csv(f) for f in csv_files], axis=0, ignore_index=True)
    print(f'Total rows: {len(df):,}')

    label_col = _detect_label_col(
        df, ['Attack_type', 'attack_type', 'Attack_Type', 'type', 'Type',
             'Label', 'label', 'category', 'attack', 'class'])
    df['label'] = df[label_col].astype(str).str.strip()
    feat = [c for c in df.columns if c not in (label_col, 'label')]
    df[feat] = df[feat].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feat)
    num_cols = df[feat].select_dtypes(include=[np.number]).columns.tolist()

    df_sub = _stratified_subset(df, 'label', TARGET_TOTAL, BENIGN_MAX) \
        if len(df) > TARGET_TOTAL else df.copy()
    del df; gc.collect()
    le = LabelEncoder()
    y = le.fit_transform(df_sub['label']).astype(np.int64)
    X = df_sub[num_cols].values.astype(np.float32)
    return _finalize(X, y, le.classes_.tolist())


# ==================================================================
# 디스패처
# ==================================================================
_LOADERS = {
    'NSL-KDD':      load_nslkdd,
    'UNSW-NB15':    load_unsw,
    'CICIDS2017':   load_cicids2017,
    'CICIDS2018':   load_cicids2018,
    'CIC-DDoS2019': load_cicddos2019,
    'Bot-IoT':      load_botiot,
    'TON_IoT':      load_toniot,
    'RT-IoT2022':   load_rtiot2022,
}


def load_network_dataset(name):
    """name 으로 데이터셋 로드. 반환:
    (X_tr, X_te, y_tr, y_te, class_counts, num_classes, class_names)"""
    if name not in _LOADERS:
        raise ValueError(f'Unknown dataset {name!r}. Available: {list(_LOADERS)}')
    return _LOADERS[name]()
