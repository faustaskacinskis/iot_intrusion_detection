"""
Modelio vertinimo modulis IoT atakų aptikimo sistemai.

Šis modulis atsakingas už:
- Modelio prognozių vertinimą
- Metrikų skaičiavimą (Accuracy, Precision, Recall, F1)
- Confusion Matrix
- ROC kreivės ir AUC
- Modelių palyginimą
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """Klasė modelių vertinimui."""
    
    def __init__(self):
        """Inicializuoja vertintoją."""
        self.evaluation_results = {}
    
    def evaluate_model(self, y_true, y_pred, model_name="Model"):
        """
        Vertina modelį su pagrindinėmis metrikomis.
        
        Args:
            y_true (array): Tikrosios etiketės
            y_pred (array): Prognozuotos etiketės
            model_name (str): Modelio pavadinimas
            
        Returns:
            dict: Vertinimo rezultatai
        """
        print(f"\n{'='*60}")
        print(f"MODELIO VERTINIMAS: {model_name}")
        print('='*60)
        
        # Skaičiuojame metrikus
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='binary', zero_division=0)
        }
        
        # Spausdiname rezultatus
        print("\nPagrindinės metrikos:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        print(f"  Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        print(f"  F1-Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
        
        # Išsaugome rezultatus
        self.evaluation_results[model_name] = metrics
        
        return metrics
    
    def get_confusion_matrix(self, y_true, y_pred, labels=None):
        """
        Skaičiuoja confusion matrix.
        
        Args:
            y_true (array): Tikrosios etiketės
            y_pred (array): Prognozuotos etiketės
            labels (list): Klasių pavadinimai
            
        Returns:
            np.array: Confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred)
        
        print("\nConfusion Matrix:")
        if labels is None:
            labels = ['Normal', 'Attack']
        
        # Sukuriame DataFrame gražesniam vaizdavimui
        cm_df = pd.DataFrame(cm, 
                            index=[f'True {l}' for l in labels],
                            columns=[f'Pred {l}' for l in labels])
        print(cm_df)
        
        # Skaičiuojame papildomus rodiklius
        if cm.shape == (2, 2):  # Binary classification
            tn, fp, fn, tp = cm.ravel()
            
            print("\nDetali analizė:")
            print(f"  True Negatives (TN):  {tn:6d}")
            print(f"  False Positives (FP): {fp:6d}")
            print(f"  False Negatives (FN): {fn:6d}")
            print(f"  True Positives (TP):  {tp:6d}")
            
            print(f"\n  False Positive Rate:  {fp/(fp+tn):.4f} ({fp/(fp+tn)*100:.2f}%)")
            print(f"  False Negative Rate:  {fn/(fn+tp):.4f} ({fn/(fn+tp)*100:.2f}%)")
        
        return cm
    
    def get_classification_report(self, y_true, y_pred, labels=None):
        """
        Generuoja detalų klasifikacijos ataskaitą.
        
        Args:
            y_true (array): Tikrosios etiketės
            y_pred (array): Prognozuotos etiketės
            labels (list): Klasių pavadinimai
            
        Returns:
            str: Klasifikacijos ataskaita
        """
        if labels is None:
            labels = ['Normal', 'Attack']
        
        print("\nDetali klasifikacijos ataskaita:")
        print(classification_report(y_true, y_pred, 
                                   target_names=labels,
                                   digits=4))
        
        return classification_report(y_true, y_pred, target_names=labels)
    
    def calculate_roc_auc(self, y_true, y_proba):
        """
        Skaičiuoja ROC kreivę ir AUC.
        
        Args:
            y_true (array): Tikrosios etiketės
            y_proba (array): Prognozuotos tikimybės (klasei 1)
            
        Returns:
            tuple: (fpr, tpr, thresholds, auc_score)
        """
        # Jei y_proba yra 2D (klasių tikimybės), imame tik antros klasės
        if len(y_proba.shape) > 1:
            y_proba = y_proba[:, 1]
        
        # Skaičiuojame ROC kreivę
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        
        # Skaičiuojame AUC
        auc_score = auc(fpr, tpr)
        
        print(f"\nROC-AUC Score: {auc_score:.4f}")
        
        return fpr, tpr, thresholds, auc_score
    
    def evaluate_multiple_models(self, models_predictions, y_true):
        """
        Vertina kelis modelius ir palygina rezultatus.
        
        Args:
            models_predictions (dict): Modelių prognozės {model_name: y_pred}
            y_true (array): Tikrosios etiketės
            
        Returns:
            pd.DataFrame: Palyginamoji lentelė
        """
        print("\n" + "="*60)
        print("KELIŲ MODELIŲ PALYGINIMAS")
        print("="*60)
        
        results = []
        
        for model_name, y_pred in models_predictions.items():
            metrics = self.evaluate_model(y_true, y_pred, model_name)
            results.append(metrics)
        
        # Sukuriame palyginamąją lentelę
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('f1_score', ascending=False)
        
        print("\n" + "="*60)
        print("REZULTATŲ SUVESTINĖ")
        print("="*60)
        print(results_df.to_string(index=False))
        
        # Nustatome geriausią modelį
        best_model = results_df.iloc[0]['model_name']
        best_f1 = results_df.iloc[0]['f1_score']
        
        print(f"\n🏆 Geriausias modelis: {best_model} (F1-Score: {best_f1:.4f})")
        
        return results_df
    
    def analyze_errors(self, y_true, y_pred, X_test=None, feature_names=None):
        """
        Analizuoja modelio klaidas.
        
        Args:
            y_true (array): Tikrosios etiketės
            y_pred (array): Prognozuotos etiketės
            X_test (array): Testavimo požymiai (optional)
            feature_names (list): Požymių pavadinimai (optional)
            
        Returns:
            dict: Klaidų analizė
        """
        print("\n" + "="*60)
        print("KLAIDŲ ANALIZĖ")
        print("="*60)
        
        # Randame False Positives ir False Negatives
        fp_mask = (y_true == 0) & (y_pred == 1)
        fn_mask = (y_true == 1) & (y_pred == 0)
        
        fp_count = np.sum(fp_mask)
        fn_count = np.sum(fn_mask)
        
        print(f"\nFalse Positives (normalus klasifikuotas kaip ataka): {fp_count}")
        print(f"False Negatives (ataka klasifikuota kaip normalus): {fn_count}")
        
        error_analysis = {
            'false_positives': fp_count,
            'false_negatives': fn_count,
            'total_errors': fp_count + fn_count
        }
        
        # Jei turime požymius, galime analizuoti klaidas giliau
        if X_test is not None and feature_names is not None:
            print("\nKlaidų požymių analizė:")
            
            # FP pavyzdžiai
            if fp_count > 0:
                fp_samples = X_test[fp_mask]
                print(f"\nFalse Positives požymių vidurkiai:")
                if hasattr(fp_samples, 'columns'):
                    for col in feature_names[:5]:  # Top 5
                        if col in fp_samples.columns:
                            print(f"  {col}: {fp_samples[col].mean():.4f}")
            
            # FN pavyzdžiai
            if fn_count > 0:
                fn_samples = X_test[fn_mask]
                print(f"\nFalse Negatives požymių vidurkiai:")
                if hasattr(fn_samples, 'columns'):
                    for col in feature_names[:5]:  # Top 5
                        if col in fn_samples.columns:
                            print(f"  {col}: {fn_samples[col].mean():.4f}")
        
        return error_analysis
    
    def evaluate_attack_types(self, y_true, y_pred, attack_types):
        """
        Vertina modelio našumą pagal atakų tipus (kelių klasių klasifikacija).
        
        Args:
            y_true (array): Tikrosios etiketės
            y_pred (array): Prognozuotos etiketės
            attack_types (list): Atakų tipų pavadinimai
            
        Returns:
            pd.DataFrame: Rezultatai pagal ataką
        """
        print("\n" + "="*60)
        print("VERTINIMAS PAGAL ATAKŲ TIPUS")
        print("="*60)
        
        results = []
        
        for i, attack in enumerate(attack_types):
            # Dvejetainė klasifikacija kiekvienai atakai
            y_true_binary = (y_true == i).astype(int)
            y_pred_binary = (y_pred == i).astype(int)
            
            metrics = {
                'attack_type': attack,
                'precision': precision_score(y_true_binary, y_pred_binary, zero_division=0),
                'recall': recall_score(y_true_binary, y_pred_binary, zero_division=0),
                'f1_score': f1_score(y_true_binary, y_pred_binary, zero_division=0),
                'support': np.sum(y_true == i)
            }
            
            results.append(metrics)
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('f1_score', ascending=False)
        
        print(results_df.to_string(index=False))
        
        # Nustatome blogiausiai aptinkamą ataką
        worst_attack = results_df.iloc[-1]
        print(f"\n⚠ Blogiausiai aptinkama ataka: {worst_attack['attack_type']} "
              f"(F1-Score: {worst_attack['f1_score']:.4f})")
        
        return results_df
    
    def assess_realtime_performance(self, inference_time, throughput):
        """
        Įvertina ar modelis tiktų realaus laiko sistemai.
        
        Args:
            inference_time (float): Vidutinis inference laikas (ms)
            throughput (int): Apdorotų pavyzdžių/s
            
        Returns:
            dict: Realaus laiko vertinimas
        """
        print("\n" + "="*60)
        print("REALAUS LAIKO SISTEMOS VERTINIMAS")
        print("="*60)
        
        # IoT tinklo reikalavimai
        max_latency_ms = 100  # Maksimali priimtina vėluotė
        min_throughput = 1000  # Minimalus pralaidumas (pavyzdžių/s)
        
        print(f"\nModelio charakteristikos:")
        print(f"  Vidutinis inference laikas: {inference_time:.2f} ms")
        print(f"  Pralaidumas: {throughput:,} pavyzdžių/s")
        
        print(f"\nIoT aplinkos reikalavimai:")
        print(f"  Maksimali latencija: {max_latency_ms} ms")
        print(f"  Minimalus pralaidumas: {min_throughput:,} pavyzdžių/s")
        
        # Vertinimas
        meets_latency = inference_time <= max_latency_ms
        meets_throughput = throughput >= min_throughput
        
        print(f"\nVertinimas:")
        print(f"  Latencijos reikalavimas: {'✓ ATITINKA' if meets_latency else '✗ NEATITINKA'}")
        print(f"  Pralaidumo reikalavimas: {'✓ ATITINKA' if meets_throughput else '✗ NEATITINKA'}")
        
        suitable = meets_latency and meets_throughput
        
        if suitable:
            print(f"\n✓ Modelis TINKAMAS realaus laiko IoT aplinkoje")
        else:
            print(f"\n✗ Modelis NETINKAMAS realaus laiko IoT aplinkoje")
            print(f"  Rekomenduojama: modelio optimizavimas arba edge computing")
        
        return {
            'inference_time_ms': inference_time,
            'throughput': throughput,
            'meets_latency': meets_latency,
            'meets_throughput': meets_throughput,
            'suitable_for_realtime': suitable
        }
    
    def export_results(self, filepath):
        """
        Eksportuoja vertinimo rezultatus į CSV.
        
        Args:
            filepath (str): Kelias kur saugoti
        """
        if not self.evaluation_results:
            print("Nėra rezultatų eksportavimui!")
            return
        
        results_df = pd.DataFrame(list(self.evaluation_results.values()))
        results_df.to_csv(filepath, index=False)
        
        print(f"✓ Rezultatai išsaugoti: {filepath}")


def main():
    """Demonstracinis pavyzdys."""
    
    print("="*60)
    print("MODELIO VERTINIMO DEMONSTRACIJA")
    print("="*60)
    
    # Generuojame sintetinius duomenis
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_classes=2,
        weights=[0.7, 0.3],
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Mokome paprastą modelį
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    # Prognozuojame
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Inicializuojame evaluatorių
    evaluator = ModelEvaluator()
    
    # Vertinimas
    metrics = evaluator.evaluate_model(y_test, y_pred, "Random Forest Demo")
    
    # Confusion Matrix
    cm = evaluator.get_confusion_matrix(y_test, y_pred)
    
    # Classification Report
    evaluator.get_classification_report(y_test, y_pred)
    
    # ROC-AUC
    fpr, tpr, thresholds, auc_score = evaluator.calculate_roc_auc(y_test, y_proba)
    
    # Klaidų analizė
    evaluator.analyze_errors(y_test, y_pred)
    
    # Realaus laiko vertinimas
    import time
    
    # Matuojame inference laiką
    start = time.time()
    for _ in range(100):
        _ = model.predict(X_test[:10])
    avg_time = ((time.time() - start) / 100) * 1000  # ms
    
    throughput = int(10 / (avg_time / 1000))
    
    evaluator.assess_realtime_performance(avg_time, throughput)
    
    print("\n✓ Demonstracija baigta")


if __name__ == "__main__":
    main()
