"""
=============================================================================
Modulis: models.py
Aprašymas: Mašininio mokymosi modelių apibrėžimas, mokymas ir vertinimas
Modeliai: Random Forest, SVM, KNN, MLP Neural Network
=============================================================================
"""

import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
import warnings
warnings.filterwarnings('ignore')


# =========================================================================
# Modelių konfigūracija
# =========================================================================

def get_models(random_state: int = 42) -> dict:
    """
    Grąžina žodyną su visais keturiais modeliais ir jų hiperparametrais.
    Hiperparametrai parinkti balansui tarp tikslumas / greitis IoT aplinkoje.
    """
    return {
        'Random Forest': RandomForestClassifier(
            n_estimators=100,      # 100 medžių – geras balansas
            max_depth=20,          # Ribojama gylis (overfitting prevencija)
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,             # Lygiagretinimas
            random_state=random_state
        ),
        'SVM': SVC(
            kernel='rbf',          # Radial basis function kernelis
            C=1.0,
            gamma='scale',
            probability=True,      # Reikalinga ROC kreivei
            random_state=random_state
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=5,         # k=5 – standartinis pasirinkimas
            metric='euclidean',
            n_jobs=-1
        ),
        'MLP Neural Network': MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),  # 3 sluoksniai
            activation='relu',
            solver='adam',
            learning_rate_init=0.001,
            max_iter=200,
            early_stopping=True,   # Stabdo mokymą kai val. loss negerėja
            validation_fraction=0.1,
            random_state=random_state
        ),
    }


# =========================================================================
# Mokymas ir vertinimas
# =========================================================================

def train_and_evaluate(model, model_name: str,
                       X_train, y_train,
                       X_test, y_test,
                       class_names: list = None) -> dict:
    """
    Apmokymas ir visapusiškas vertinimas vieno modelio.

    Parametrai:
        model       – sklearn modelio objektas
        model_name  – modelio pavadinimas (žurnalui)
        X_train/y_train – mokymo duomenys
        X_test/y_test   – testavimo duomenys
        class_names – klasių pavadinimai (daugiaklasei klasifikacijai)

    Grąžina žodyną su metrikomis ir modelio objektu.
    """
    print(f"\n{'='*60}")
    print(f"  Modelis: {model_name}")
    print(f"{'='*60}")

    # --- Mokymas ---
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"  Mokymo laikas: {train_time:.2f}s")

    # --- Prognozavimas ---
    t0 = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - t0
    print(f"  Prognozavimo laikas: {inference_time*1000:.1f}ms ({len(y_test):,} įrašų)")

    # --- Pagrindinės metrikos ---
    is_binary = len(np.unique(y_test)) == 2
    avg = 'binary' if is_binary else 'weighted'

    acc       = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average=avg, zero_division=0)
    recall    = recall_score(y_test, y_pred, average=avg, zero_division=0)
    f1        = f1_score(y_test, y_pred, average=avg, zero_division=0)
    cm        = confusion_matrix(y_test, y_pred)

    print(f"\n  Accuracy  : {acc:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1-score  : {f1:.4f}")

    # --- ROC / AUC (tik dvejetainei) ---
    roc_auc = None
    fpr, tpr, thresholds = None, None, None
    if is_binary and hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        print(f"  ROC AUC   : {roc_auc:.4f}")

    # --- Detalus klasifikacijos ataskaita ---
    print(f"\n  Klasifikacijos ataskaita:")
    print(classification_report(
        y_test, y_pred,
        target_names=class_names,
        zero_division=0
    ))

    return {
        'model':          model,
        'model_name':     model_name,
        'y_pred':         y_pred,
        'accuracy':       acc,
        'precision':      precision,
        'recall':         recall,
        'f1':             f1,
        'roc_auc':        roc_auc,
        'fpr':            fpr,
        'tpr':            tpr,
        'confusion_matrix': cm,
        'train_time':     train_time,
        'inference_time': inference_time,
        'class_names':    class_names,
    }


def train_all_models(data: dict, mode: str = 'binary', random_state: int = 42) -> dict:
    """
    Apmokymas ir vertinimas visų keturių modelių.

    Parametrai:
        data        – žodynas iš prepare_datasets()
        mode        – 'binary' arba 'multiclass'
        random_state – atsitiktinumo sėkla

    Grąžina žodyną su kiekvieno modelio rezultatais.
    """
    X_train = data['X_train']
    X_test  = data['X_test']

    if mode == 'binary':
        y_train     = data['y_bin_train']
        y_test      = data['y_bin_test']
        class_names = ['Normal', 'Attack']
    else:
        y_train     = data['y_multi_train']
        y_test      = data['y_multi_test']
        le          = data['label_encoder']
        class_names = list(le.classes_)

    print(f"\n{'#'*60}")
    print(f"  REŽIMAS: {'Dvejetainė' if mode == 'binary' else 'Daugiaklasei'} klasifikacija")
    print(f"  Klasės: {class_names}")
    print(f"{'#'*60}")

    # SVM lėtas su dideliais duomenimis – naudojame poaimį
    svm_limit = 15000
    models = get_models(random_state)
    results = {}

    for name, model in models.items():
        if name == 'SVM' and len(X_train) > svm_limit:
            print(f"\n[ĮSPĖJIMAS] SVM – naudojamas {svm_limit:,} įrašų poaibis (greičiui)")
            idx = np.random.choice(len(X_train), svm_limit, replace=False)
            _X_train = X_train[idx]
            _y_train = y_train[idx]
        else:
            _X_train = X_train
            _y_train = y_train

        results[name] = train_and_evaluate(
            model, name, _X_train, _y_train, X_test, y_test, class_names
        )

    return results


def get_feature_importance(results: dict, feature_names: list) -> dict:
    """
    Ištraukia požymių svarbą iš Random Forest modelio.

    Grąžina žodyną: požymio_pavadinimas -> svarba (0–1)
    """
    rf_result = results.get('Random Forest')
    if rf_result is None:
        return {}

    rf_model      = rf_result['model']
    importances   = rf_model.feature_importances_
    indices       = np.argsort(importances)[::-1]

    importance_dict = {
        feature_names[i]: float(importances[i])
        for i in indices
    }
    return importance_dict
