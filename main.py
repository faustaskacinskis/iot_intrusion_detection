"""
Pagrindinis vykdomasis failas IoT atakų aptikimo sistemai.

Šis failas integruoja visus modulius ir vykdo pilną pipeline:
1. Duomenų paruošimą
2. Modelių mokymąsi
3. Vertinimą
4. Vizualizaciją
"""

import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Importuojame savo modulius
from data_preprocessing import DataPreprocessor
from model_training import ModelTrainer
from model_evaluation import ModelEvaluator
from visualization import Visualizer


def create_directories():
    """Sukuria reikalingas direktorijas jei jos neegzistuoja."""
    dirs = ['data', 'models', 'results', 'results/plots']
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)


def generate_demo_data(filepath, n_samples=5000):
    """
    Generuoja sintetinius IoT tinklo duomenis demonstracijai.
    
    Args:
        filepath (str): Kelias kur saugoti
        n_samples (int): Pavyzdžių skaičius
    """
    print("\n" + "="*60)
    print("GENERUOJAMI SINTETINIAI DUOMENYS")
    print("="*60)
    
    np.random.seed(42)
    
    # Simuliuojame IoT tinklo požymius
    data = {
        # Tinklo charakteristikos
        'duration': np.random.exponential(5, n_samples),  # Srauto trukmė (s)
        'protocol': np.random.choice(['TCP', 'UDP', 'ICMP'], n_samples, p=[0.6, 0.3, 0.1]),
        'src_port': np.random.randint(1024, 65535, n_samples),
        'dst_port': np.random.choice([80, 443, 22, 21, 3389, 8080], n_samples),
        
        # Srauto statistikos
        'packets': np.random.poisson(50, n_samples),
        'bytes': np.random.exponential(1000, n_samples),
        'rate': np.random.uniform(0, 100, n_samples),  # Paketų/s
        
        # Paketų charakteristikos
        'packet_size_avg': np.random.normal(500, 200, n_samples),
        'packet_size_std': np.random.exponential(100, n_samples),
        
        # Laiko charakteristikos
        'inter_arrival_time': np.random.exponential(0.1, n_samples),
        'flow_duration': np.random.exponential(10, n_samples),
        
        # TCP flags (simuliacija)
        'syn_count': np.random.poisson(2, n_samples),
        'ack_count': np.random.poisson(10, n_samples),
        'fin_count': np.random.poisson(1, n_samples),
        
        # IoT specifiniai požymiai
        'device_type': np.random.choice(['camera', 'sensor', 'gateway', 'controller'], 
                                       n_samples, p=[0.3, 0.4, 0.2, 0.1]),
    }
    
    # Tikslinė klasė
    # 70% normalus, 30% atakos
    labels = np.random.choice(['normal', 'attack'], n_samples, p=[0.7, 0.3])
    
    # Pridedame atakų požymius
    attack_indices = np.where(labels == 'attack')[0]
    
    # DoS atakos: dideli paketų skaičiai
    dos_indices = attack_indices[:len(attack_indices)//3]
    data['packets'] = np.array(data['packets'], dtype=float)  # Konvertuojame į float
    data['rate'] = np.array(data['rate'], dtype=float)
    data['packets'][dos_indices] *= np.random.uniform(10, 100, len(dos_indices))
    data['rate'][dos_indices] *= np.random.uniform(5, 50, len(dos_indices))
    
    # Port scanning: daug unikalių portų
    scan_indices = attack_indices[len(attack_indices)//3:2*len(attack_indices)//3]
    data['dst_port'][scan_indices] = np.random.randint(1, 65535, len(scan_indices))
    
    data['label'] = labels
    
    # Sukuriame DataFrame
    df = pd.DataFrame(data)
    
    # Užtikriname, kad nėra neigiamų reikšmių
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].abs()
    
    # Išsaugome
    df.to_csv(filepath, index=False)
    
    print(f"✓ Sugeneruota {n_samples} pavyzdžių")
    print(f"✓ Požymių skaičius: {len(df.columns) - 1}")
    print(f"✓ Klasių pasiskirstymas:")
    print(df['label'].value_counts())
    print(f"✓ Išsaugota: {filepath}")


def run_full_pipeline(data_path, models_to_train='all', binary=True, test_size=0.2):
    """
    Vykdo pilną ML pipeline nuo duomenų iki rezultatų.
    
    Args:
        data_path (str): Kelias iki duomenų failo
        models_to_train (str or list): Kokie modeliai mokyti ('all' arba sąrašas)
        binary (bool): Dvejetainė klasifikacija
        test_size (float): Testavimo aibės dalis
        
    Returns:
        dict: Rezultatai
    """
    start_time = time.time()
    
    print("\n" + "="*60)
    print("PRADEDAMAS PILNAS ML PIPELINE")
    print("="*60)
    
    # ========================================
    # 1. DUOMENŲ PARUOŠIMAS
    # ========================================
    print("\n### ŽINGSnis 1: DUOMENŲ PARUOŠIMAS ###")
    
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(
        data_path,
        binary_classification=binary,
        test_size=test_size
    )
    
    feature_names = preprocessor.feature_names
    
    # ========================================
    # 2. MODELIŲ MOKYMAS
    # ========================================
    print("\n### ŽINGSNIS 2: MODELIŲ MOKYMAS ###")
    
    trainer = ModelTrainer()
    trainer.initialize_models()
    
    # Nustatome kuriuos modelius mokyti
    if models_to_train == 'all':
        models_list = list(trainer.models.keys())
    else:
        models_list = models_to_train if isinstance(models_to_train, list) else [models_to_train]
    
    # Mokome tik pasirinktus modelius
    for model_name in models_list:
        trainer.train_model(model_name, X_train, y_train)
    
    # ========================================
    # 3. VERTINIMAS
    # ========================================
    print("\n### ŽINGSNIS 3: MODELIŲ VERTINIMAS ###")
    
    evaluator = ModelEvaluator()
    
    # Surenkame prognozės iš visų modelių
    predictions = {}
    probabilities = {}
    
    for model_name in models_list:
        y_pred = trainer.predict(model_name, X_test)
        predictions[model_name] = y_pred
        
        # Tikimybės (jei modelis palaiko)
        try:
            y_proba = trainer.predict_proba(model_name, X_test)
            probabilities[model_name] = y_proba
        except:
            probabilities[model_name] = None
    
    # Vertiname kiekvieną modelį
    for model_name, y_pred in predictions.items():
        print(f"\n{'-'*60}")
        metrics = evaluator.evaluate_model(y_test, y_pred, model_name)
        
        # Confusion Matrix
        cm = evaluator.get_confusion_matrix(y_test, y_pred)
        
        # Classification Report
        evaluator.get_classification_report(y_test, y_pred)
        
        # ROC-AUC (jei galima)
        if probabilities[model_name] is not None:
            fpr, tpr, thresholds, auc_score = evaluator.calculate_roc_auc(
                y_test, probabilities[model_name]
            )
        
        # Klaidų analizė
        evaluator.analyze_errors(y_test, y_pred, X_test, feature_names)
    
    # Palyginame visus modelius
    results_df = evaluator.evaluate_multiple_models(predictions, y_test)
    
    # ========================================
    # 4. VIZUALIZACIJA
    # ========================================
    print("\n### ŽINGSNIS 4: REZULTATŲ VIZUALIZACIJA ###")
    
    viz = Visualizer()
    
    # 4.1 Klasių pasiskirstymas
    viz.plot_class_distribution(
        y_test,
        labels=['Normal', 'Attack'],
        title="Testavimo aibės klasių pasiskirstymas",
        save_path='results/plots/class_distribution.png'
    )
    
    # 4.2 Modelių palyginimas
    viz.plot_model_comparison(
        results_df,
        save_path='results/plots/model_comparison.png'
    )
    
    # 4.3 Confusion Matrix (geriausiui modeliui)
    best_model = results_df.iloc[0]['model_name']
    best_cm = evaluator.get_confusion_matrix(y_test, predictions[best_model])
    
    viz.plot_confusion_matrix(
        best_cm,
        labels=['Normal', 'Attack'],
        title=f"Confusion Matrix - {best_model}",
        save_path='results/plots/confusion_matrix_best.png'
    )
    
    # 4.4 ROC kreivės
    roc_data = {}
    for model_name in models_list:
        if probabilities[model_name] is not None:
            fpr, tpr, _, auc_score = evaluator.calculate_roc_auc(
                y_test, probabilities[model_name]
            )
            roc_data[model_name] = (fpr, tpr, auc_score)
    
    if roc_data:
        viz.plot_multiple_roc_curves(
            roc_data,
            save_path='results/plots/roc_curves.png'
        )
    
    # 4.5 Požymių svarba (Random Forest)
    if 'Random Forest' in trainer.trained_models:
        importance_df = trainer.get_feature_importance('Random Forest', feature_names)
        if importance_df is not None:
            viz.plot_feature_importance(
                importance_df,
                top_n=15,
                save_path='results/plots/feature_importance.png'
            )
    
    # 4.6 Mokymo laikų palyginimas
    viz.plot_training_time_comparison(
        trainer.training_times,
        save_path='results/plots/training_times.png'
    )
    
    # ========================================
    # 5. REALAUS LAIKO VERTINIMAS
    # ========================================
    print("\n### ŽINGSNIS 5: REALAUS LAIKO NAŠUMO VERTINIMAS ###")
    
    # Matuojame inference laiką (Random Forest)
    if 'Random Forest' in trainer.trained_models:
        n_iterations = 100
        sample_size = 100
        
        inference_start = time.time()
        for _ in range(n_iterations):
            _ = trainer.predict('Random Forest', X_test[:sample_size])
        inference_time_ms = ((time.time() - inference_start) / n_iterations) * 1000
        
        throughput = int(sample_size / (inference_time_ms / 1000))
        
        evaluator.assess_realtime_performance(inference_time_ms, throughput)
    
    # ========================================
    # 6. MODELIŲ IŠSAUGOJIMAS
    # ========================================
    print("\n### ŽINGSNIS 6: MODELIŲ IŠSAUGOJIMAS ###")
    
    for model_name in models_list:
        model_path = f"models/{model_name.lower().replace(' ', '_')}.pkl"
        trainer.save_model(model_name, model_path)
    
    # Išsaugome preprocessor
    import joblib
    joblib.dump(preprocessor, 'models/preprocessor.pkl')
    print("✓ Preprocessor išsaugotas: models/preprocessor.pkl")
    
    # Išsaugome rezultatus
    results_df.to_csv('results/model_results.csv', index=False)
    print("✓ Rezultatai išsaugoti: results/model_results.csv")
    
    # ========================================
    # BAIGIAMOJI SUVESTINĖ
    # ========================================
    total_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("PIPELINE BAIGTAS")
    print("="*60)
    print(f"Bendras laikas: {total_time:.2f} sekundės ({total_time/60:.2f} minutės)")
    print(f"Išmokyti modeliai: {len(models_list)}")
    print(f"Geriausias modelis: {best_model}")
    print(f"Geriausias F1-Score: {results_df.iloc[0]['f1_score']:.4f}")
    print("\nVisi rezultatai išsaugoti 'results/' direktorijoje")
    print("="*60)
    
    return {
        'preprocessor': preprocessor,
        'trainer': trainer,
        'evaluator': evaluator,
        'results_df': results_df,
        'best_model': best_model
    }


def main():
    """Pagrindinis vykdymas su argumentų apdorojimu."""
    
    parser = argparse.ArgumentParser(
        description='IoT Tinklo Atakų Aptikimo Sistema su Mašininiu Mokymusi'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        default='data/iot_network_data.csv',
        help='Kelias iki duomenų failo'
    )
    
    parser.add_argument(
        '--models',
        type=str,
        default='all',
        help='Modeliai mokymui: "all", "random_forest", "svm", "knn", "neural_network"'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Testavimo aibės dalis (0-1)'
    )
    
    parser.add_argument(
        '--generate-demo',
        action='store_true',
        help='Generuoti sintetinius demo duomenis'
    )
    
    parser.add_argument(
        '--demo-samples',
        type=int,
        default=5000,
        help='Sintetinių duomenų kiekis'
    )
    
    args = parser.parse_args()
    
    # Sukuriame direktorijas
    create_directories()
    
    # Jei prašoma, generuojame demo duomenis
    if args.generate_demo or not os.path.exists(args.data):
        print("Duomenų failas nerastas. Generuojami sintetiniai duomenys...")
        generate_demo_data(args.data, args.demo_samples)
    
    # Apdorojame modelių sąrašą
    if args.models == 'all':
        models = 'all'
    else:
        model_map = {
            'random_forest': 'Random Forest',
            'svm': 'SVM',
            'knn': 'KNN',
            'neural_network': 'Neural Network'
        }
        models = [model_map.get(m.strip(), m.strip()) for m in args.models.split(',')]
    
    # Vykdome pipeline
    results = run_full_pipeline(
        data_path=args.data,
        models_to_train=models,
        binary=True,
        test_size=args.test_size
    )
    
    print("\n✓ Programa baigta sėkmingai!")
    print("Rezultatai pasiekiami 'results/' direktorijoje")


if __name__ == "__main__":
    main()
