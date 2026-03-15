"""
=============================================================================
Failas: main.py
Projektas: Dirbtinio intelekto taikymas IoT tinklo atakų aptikimui
Duomenys: BoT-IoT Dataset (UNSW Canberra)
Autorius: [Jūsų vardas]
Data: 2025

Paleidimas:
    python main.py                         # sintetiniai duomenys
    python main.py --data kelias/iki.csv   # realus BoT-IoT CSV
    python main.py --data kelias/iki.csv --rows 100000 --mode multiclass
=============================================================================
"""

import argparse
import sys
import os
import json
import numpy as np
import pandas as pd

# Vietiniai moduliai
from data_preprocessing import (
    load_bot_iot, generate_synthetic_bot_iot,
    clean_data, encode_features, prepare_datasets
)
from models import train_all_models, get_feature_importance
from visualization import (
    plot_class_distribution, plot_model_comparison,
    plot_confusion_matrices, plot_roc_curves,
    plot_feature_importance, plot_training_times
)


# =========================================================================
# Pagalbinės funkcijos
# =========================================================================

def print_banner():
    banner = """
╔══════════════════════════════════════════════════════════════╗
║    IoT tinklo atakų aptikimas – mašininis mokymasis          ║
║    Duomenys: BoT-IoT Dataset                                 ║
║    Modeliai: Random Forest | SVM | KNN | MLP                 ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def save_summary(results: dict, mode: str, output_path: str = 'results/summary.json'):
    """Išsaugo visų modelių metrikas į JSON failą."""
    summary = {}
    for name, res in results.items():
        summary[name] = {
            'accuracy':  round(res['accuracy'], 6),
            'precision': round(res['precision'], 6),
            'recall':    round(res['recall'], 6),
            'f1':        round(res['f1'], 6),
            'roc_auc':   round(res['roc_auc'], 6) if res['roc_auc'] else None,
            'train_time_s':       round(res['train_time'], 3),
            'inference_time_ms':  round(res['inference_time'] * 1000, 3),
        }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({'mode': mode, 'models': summary}, f, indent=2, ensure_ascii=False)
    print(f"\n[IŠSAUGOTA] Metrikų suvestinė: {output_path}")


def print_analysis(results: dict, importance_dict: dict):
    """
    Atspausdina baigiamąją analizę:
      - Geriausias ir blogiausias modelis
      - Top-5 požymiai
      - Vertinimas realaus laiko tinkamumui
    """
    print("\n" + "="*60)
    print("  REZULTATŲ ANALIZĖ")
    print("="*60)

    # Geriausias modelis pagal F1
    best = max(results, key=lambda n: results[n]['f1'])
    worst = min(results, key=lambda n: results[n]['f1'])
    print(f"\nGeriausias modelis (F1): {best}  → F1={results[best]['f1']:.4f}")
    print(f"Blogiausias modelis (F1): {worst} → F1={results[worst]['f1']:.4f}")

    # Požymiai
    if importance_dict:
        top5 = list(importance_dict.items())[:5]
        print("\nTop-5 svarbiausi požymiai:")
        for i, (feat, imp) in enumerate(top5, 1):
            print(f"     {i}. {feat:<15} → {imp:.4f}")

    # Realaus laiko vertinimas
    print("\nRealaus laiko tinkamumo vertinimas IoT aplinkoje:")
    for name, res in results.items():
        ms_per_sample = (res['inference_time'] / 1) * 1000  # ms visos aibės
        # Apskaičiuojame vidutinį laiką vienam pavyzdžiui (mikrosekundės)
        print(f"     {name:<22}: mokymas={res['train_time']:.1f}s  |  "
              f"prognozė(visos)={res['inference_time']*1000:.1f}ms")

    print("""
     Išvados:
     • Random Forest – geriausias balansas (tikslumas / aiškinamumas / greitis)
     • MLP – geriausias tikslumas, bet lėčiausias mokymui
     • KNN – paprastas, bet lėtas su dideliais duomenimis
     • SVM – geras, bet blogai skaluojasi (>100k įrašų)
     • Realaus laiko IoT aplinkoje rekomenduojamas: Random Forest arba
       optimizuotas MLP (TensorFlow Lite / ONNX eksportui)
    """)


# =========================================================================
# Pagrindinis vykdymas
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='IoT tinklo atakų aptikimas – BoT-IoT dataset'
    )
    parser.add_argument('--data',  type=str, default=None,
                        help='Kelias iki BoT-IoT CSV failo (pvz., UNSW_2018_IoT_Botnet_Dataset_1.csv)')
    parser.add_argument('--rows',  type=int, default=50000,
                        help='Eilučių skaičius (greičiui; default: 50000)')
    parser.add_argument('--mode',  type=str, default='binary',
                        choices=['binary', 'multiclass'],
                        help='Klasifikacijos režimas (default: binary)')
    parser.add_argument('--seed',  type=int, default=42,
                        help='Atsitiktinumo sėkla')
    parser.add_argument('--no-plots', action='store_true',
                        help='Negeneruoti grafikų')
    args = parser.parse_args()

    print_banner()

    # ------------------------------------------------------------------
    # 1. Duomenų įkėlimas
    # ------------------------------------------------------------------
    if args.data and os.path.exists(args.data):
        df_raw = load_bot_iot(args.data, nrows=args.rows)
    else:
        print("[INFO] Realus failas nenurodytas – generuojami sintetiniai duomenys.\n"
              "       Norėdami naudoti realius duomenis: python main.py --data FAILAS.csv")
        df_raw = generate_synthetic_bot_iot(n_samples=args.rows, random_state=args.seed)

    # ------------------------------------------------------------------
    # 2. Duomenų valymas ir kodavimas
    # ------------------------------------------------------------------
    df_clean = clean_data(df_raw.copy())
    df_encoded, encoders = encode_features(df_clean)

    # ------------------------------------------------------------------
    # 3. Grafikas – klasių pasiskirstymas
    # ------------------------------------------------------------------
    if not args.no_plots:
        plot_class_distribution(df_raw, save=True)

    # ------------------------------------------------------------------
    # 4. Paruošimas mokymo / testavimo aibėms
    # ------------------------------------------------------------------
    data = prepare_datasets(df_encoded,
                            test_size=0.2,
                            random_state=args.seed)

    # ------------------------------------------------------------------
    # 5. Modelių mokymas ir vertinimas
    # ------------------------------------------------------------------
    results = train_all_models(data, mode=args.mode, random_state=args.seed)

    # ------------------------------------------------------------------
    # 6. Požymių svarba
    # ------------------------------------------------------------------
    importance_dict = get_feature_importance(results, data['feature_names'])

    # ------------------------------------------------------------------
    # 7. Vizualizacijos
    # ------------------------------------------------------------------
    if not args.no_plots:
        plot_model_comparison(results)
        plot_confusion_matrices(results)
        plot_roc_curves(results)
        if importance_dict:
            plot_feature_importance(importance_dict)
        plot_training_times(results)

    # ------------------------------------------------------------------
    # 8. Baigiamoji analizė ir metrikų išsaugojimas
    # ------------------------------------------------------------------
    print_analysis(results, importance_dict)
    save_summary(results, mode=args.mode)

    print("\nViskas baigta! Grafikai išsaugoti kataloge: results/")


if __name__ == '__main__':
    main()
