"""
=============================================================================
Modulis: data_preprocessing.py
Aprašymas: BoT-IoT duomenų rinkinio įkėlimas, valymas ir paruošimas
=============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# -------------------------------------------------------------------------
# BoT-IoT duomenų rinkinio požymių sąrašas (UNSW pagal oficialią dokumentaciją)
# -------------------------------------------------------------------------
BOT_IOT_COLUMNS = [
    'pkSeqID', 'stime', 'flgs', 'proto', 'saddr', 'sport', 'daddr', 'dport',
    'pkts', 'bytes', 'state', 'ltime', 'seq', 'dur', 'mean', 'stddev',
    'smac', 'dmac', 'sum', 'min', 'max', 'soui', 'doui', 'sco', 'dco',
    'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'srate', 'drate',
    'attack', 'category', 'subcategory'
]

# Požymiai, kurie bus pašalinti (identifikatoriai, MAC adresai ir kt.)
DROP_FEATURES = ['pkSeqID', 'stime', 'ltime', 'smac', 'dmac',
                 'soui', 'doui', 'sco', 'dco', 'saddr', 'daddr',
                 'sport', 'dport', 'seq']

# Kategoriniai požymiai, kuriuos reikia koduoti
CATEGORICAL_FEATURES = ['proto', 'state', 'flgs']


def load_bot_iot(filepath: str, nrows: int = None) -> pd.DataFrame:
    """
    Įkelia BoT-IoT CSV failą.
    Automatiškai aptinka, ar failas turi antraštę, ar ne.

    Parametrai:
        filepath : str  – kelias iki CSV failo
        nrows    : int  – eilučių skaičius (None = visos)

    Grąžina:
        pd.DataFrame su pavadintais stulpeliais
    """
    print(f"[INFO] Įkeliami duomenys iš: {filepath}")

    # Patikriname pirmą eilutę – ar tai antraštė ar duomenys?
    first_row = pd.read_csv(filepath, nrows=1, header=None).iloc[0]
    first_val = str(first_row[0]).strip()

    # Jei pirmoji reikšmė yra skaičius – failas neturi antraštės
    has_header = not first_val.lstrip('-').replace('.', '', 1).isdigit()

    if has_header:
        df = pd.read_csv(filepath, nrows=nrows, low_memory=False)
        print(f"[INFO] Aptikta antraštės eilutė – naudojami originalūs pavadinimai")
    else:
        df = pd.read_csv(filepath, nrows=nrows, low_memory=False, header=None)
        df.columns = BOT_IOT_COLUMNS[:len(df.columns)]
        print(f"[INFO] Antraštė nerasta – priskirti BoT-IoT stulpelių pavadinimai")

    print(f"[INFO] Įkelta eilučių: {len(df):,}, stulpelių: {df.shape[1]}")
    print(f"[INFO] Stulpeliai: {list(df.columns)}")
    return df


def generate_synthetic_bot_iot(n_samples: int = 50000, random_state: int = 42) -> pd.DataFrame:
    """
    Generuoja sintetinius BoT-IoT tipo duomenis demonstracijai,
    kai realus duomenų rinkinys nepasiekiamas.

    Atakų klasės ir jų tikimybės atspindi realų BoT-IoT pasiskirstymą:
      - Normal          ~15%
      - DoS             ~35%
      - DDoS            ~30%
      - Reconnaissance  ~12%
      - Theft            ~8%
    """
    print(f"[INFO] Generuojami {n_samples:,} sintetiniai BoT-IoT įrašai...")
    rng = np.random.default_rng(random_state)

    categories = ['Normal', 'DoS', 'DDoS', 'Reconnaissance', 'Theft']
    weights     = [0.15,    0.35,  0.30,   0.12,             0.08]
    cat_labels  = rng.choice(categories, size=n_samples, p=weights)

    def _vals(cat):
        """Grąžina kiekvieno požymio reikšmių intervalus pagal atakos tipą."""
        base = {
            'Normal':         dict(pkts=(1,50),    bytes=(64,1500),   dur=(0.01,10),   rate=(1,100)),
            'DoS':            dict(pkts=(100,5000), bytes=(40,100),    dur=(0.001,0.5), rate=(5000,50000)),
            'DDoS':           dict(pkts=(200,8000), bytes=(40,100),    dur=(0.0001,0.1),rate=(10000,100000)),
            'Reconnaissance': dict(pkts=(1,10),    bytes=(40,200),    dur=(0.001,1),   rate=(1,50)),
            'Theft':          dict(pkts=(10,200),  bytes=(200,5000),  dur=(0.1,30),    rate=(10,500)),
        }
        return base[cat]

    rows = []
    for cat in cat_labels:
        v = _vals(cat)
        pkts  = rng.integers(*v['pkts'])
        byt   = rng.integers(*v['bytes']) * pkts
        dur   = rng.uniform(*v['dur'])
        rate  = rng.uniform(*v['rate'])
        spkts = int(pkts * rng.uniform(0.3, 0.7))
        dpkts = pkts - spkts
        sbytes = int(byt * rng.uniform(0.2, 0.8))
        dbytes = byt - sbytes

        rows.append({
            'proto':    rng.choice(['tcp', 'udp', 'icmp', 'arp'], p=[0.5,0.3,0.15,0.05]),
            'state':    rng.choice(['FIN', 'CON', 'RST', 'REQ', 'INT'], p=[0.3,0.3,0.2,0.1,0.1]),
            'flgs':     rng.choice(['e s', 'e', '   ', 'e g'], p=[0.4,0.3,0.2,0.1]),
            'pkts':     pkts,
            'bytes':    byt,
            'dur':      dur,
            'mean':     byt / max(pkts, 1),
            'stddev':   rng.uniform(0, 200),
            'sum':      byt * 2,
            'min':      rng.integers(40, 100),
            'max':      rng.integers(100, 1500),
            'spkts':    spkts,
            'dpkts':    dpkts,
            'sbytes':   sbytes,
            'dbytes':   dbytes,
            'rate':     rate,
            'srate':    rate * rng.uniform(0.3, 0.7),
            'drate':    rate * rng.uniform(0.3, 0.7),
            'attack':   0 if cat == 'Normal' else 1,
            'category': cat,
            'subcategory': cat,
        })

    df = pd.DataFrame(rows)
    print(f"[INFO] Sugeneruota. Klasių pasiskirstymas:\n{df['category'].value_counts().to_string()}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Valo duomenis:
      - Pašalina identifikacinius ir nesvarbius stulpelius
      - Keičia begalybes ir NaN į medianas
      - Pašalina pilnai besidubliuojančias eilutes
    """
    print("[INFO] Pradedamas duomenų valymas...")

    # Pašalinami nereikalingi stulpeliai (jei egzistuoja)
    drop_cols = [c for c in DROP_FEATURES if c in df.columns]
    df = df.drop(columns=drop_cols)
    print(f"[INFO] Pašalinti stulpeliai: {drop_cols}")

    # Pakeičiamos begalinės reikšmės į NaN, tada užpildomos mediana
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        median = df[col].median()
        df[col].fillna(median, inplace=True)

    # Šalinami besidubliuojantys įrašai
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"[INFO] Pašalinta {before - len(df):,} besidubliuojančių eilučių")

    return df


def encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Koduoja kategorinius požymius naudojant LabelEncoder.

    Grąžina:
        df          – transformuotas DataFrame
        encoders    – žodynas su kiekvieno stulpelio LabelEncoder objektu
    """
    print("[INFO] Koduojami kategoriniai požymiai...")
    encoders = {}
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        print(f"  {col}: {list(le.classes_)}")

    return df, encoders


def prepare_datasets(df: pd.DataFrame,
                     target_col: str = 'attack',
                     multiclass_col: str = 'category',
                     test_size: float = 0.2,
                     random_state: int = 42):
    print("[INFO] Paruošiamos mokymo / testavimo aibės...")

    # Jei 'attack' stulpelio nėra – ieškome alternatyvų
    for alt in [target_col, 'label', 'Label', 'class']:
        if alt in df.columns:
            target_col = alt
            break
    else:
        raise ValueError(f"Nerasta jokia etiketės kolona! Turimi stulpeliai: {list(df.columns)}")

    # Jei 'category' stulpelio nėra – naudojame target_col kaip atsarginę
    if multiclass_col not in df.columns:
        for alt in ['subcategory', 'attack_type', 'Category']:
            if alt in df.columns:
                multiclass_col = alt
                print(f"[INFO] Daugiaklasei klasifikacijai naudojamas stulpelis: '{multiclass_col}'")
                break
        else:
            print(f"[ĮSPĖJIMAS] Stulpelis '{multiclass_col}' nerastas – daugiaklasei bus naudojama '{target_col}'")
            multiclass_col = target_col
    """
    Paruošia mokymo ir testavimo aibes dvejetainei bei daugiaklasei klasifikacijai.

    Parametrai:
        df             – išvalytas ir užkoduotas DataFrame
        target_col     – dvejetainės klasifikacijos etiketė ('attack')
        multiclass_col – daugiaklasei klasifikacijai ('category')
        test_size      – testavimo dalies dydis
        random_state   – atsitiktinumo sėkla

    Grąžina:
        Žodynas su X_train, X_test, y_bin_train, y_bin_test,
        y_multi_train, y_multi_test, feature_names, scaler, label_encoder
    """
    # Požymių matrica (be etikečių stulpelių)
    drop_targets = [c for c in [target_col, multiclass_col, 'subcategory'] if c in df.columns]
    X = df.drop(columns=drop_targets)
    feature_names = list(X.columns)

    # Dvejetainės etiketės
    y_binary = df[target_col].values if target_col in df.columns else None

    # Daugiaklasei – koduojame kategorijų pavadinimus
    le_multi = LabelEncoder()
    y_multi = le_multi.fit_transform(df[multiclass_col].astype(str)) \
        if multiclass_col in df.columns else None

    # Skaidymas
    X_train, X_test, \
    y_bin_train, y_bin_test, \
    y_multi_train, y_multi_test = train_test_split(
        X, y_binary, y_multi,
        test_size=test_size,
        random_state=random_state,
        stratify=y_binary
    )

    # Normalizavimas (StandardScaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    print(f"[INFO] Mokymo aibė : {X_train_scaled.shape[0]:,} įrašų")
    print(f"[INFO] Testavimo aibė: {X_test_scaled.shape[0]:,} įrašų")
    print(f"[INFO] Požymių skaičius: {len(feature_names)}")

    return {
        'X_train': X_train_scaled,
        'X_test':  X_test_scaled,
        'y_bin_train':   y_bin_train,
        'y_bin_test':    y_bin_test,
        'y_multi_train': y_multi_train,
        'y_multi_test':  y_multi_test,
        'feature_names': feature_names,
        'scaler':        scaler,
        'label_encoder': le_multi,
    }
