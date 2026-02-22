"""
Modelio mokymo modulis IoT atakų aptikimo sistemai.

Šis modulis įgyvendina kelis mašininio mokymosi algoritmus:
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Neural Network (MLP)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import joblib
import time
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """Klasė modelių mokymui ir valdymui."""
    
    def __init__(self):
        """Inicializuoja modelių trainerį."""
        self.models = {}
        self.training_times = {}
        self.trained_models = {}
        
    def initialize_models(self):
        """Inicializuoja visus mašininio mokymosi modelius."""
        
        print("\nInicializuojami modeliai...")
        
        # 1. Random Forest - Pagrindinis modelis
        self.models['Random Forest'] = RandomForestClassifier(
            n_estimators=100,           # Medžių skaičius
            max_depth=20,               # Maksimalus gylis
            min_samples_split=10,       # Min. pavyzdžių dalybai
            min_samples_leaf=4,         # Min. pavyzdžių lape
            max_features='sqrt',        # Požymių skaičius kiekv. dalybai
            random_state=42,
            n_jobs=-1,                  # Naudoti visus CPU
            verbose=0
        )
        
        # 2. Support Vector Machine - Palyginimui
        self.models['SVM'] = SVC(
            kernel='rbf',               # Radial basis function kernel
            C=1.0,                      # Regularizacijos parametras
            gamma='scale',              # Kernel koeficientas
            random_state=42,
            verbose=False
        )
        
        # 3. K-Nearest Neighbors - Bazinis modelis
        self.models['KNN'] = KNeighborsClassifier(
            n_neighbors=5,              # Kaimynų skaičius
            weights='distance',         # Svoris pagal atstumą
            algorithm='auto',           # Algoritmo pasirinkimas
            n_jobs=-1
        )
        
        # 4. Neural Network (MLP) - Pažangus modelis
        self.models['Neural Network'] = MLPClassifier(
            hidden_layer_sizes=(64, 32, 16),  # 3 paslėpti sluoksniai
            activation='relu',          # ReLU aktyvavimo funkcija
            solver='adam',              # Adam optimizatorius
            alpha=0.0001,              # L2 regularizacija
            batch_size=128,
            learning_rate='adaptive',
            max_iter=200,
            random_state=42,
            verbose=False,
            early_stopping=True,        # Ankstyvasis sustabdymas
            validation_fraction=0.1
        )
        
        print(f"✓ Inicializuoti {len(self.models)} modeliai:")
        for name in self.models.keys():
            print(f"  - {name}")
    
    def train_model(self, model_name, X_train, y_train, verbose=True):
        """
        Mokosi vieną modelį.
        
        Args:
            model_name (str): Modelio pavadinimas
            X_train (array): Mokymo požymiai
            y_train (array): Mokymo etiketės
            verbose (bool): Ar spausdinti progresą
            
        Returns:
            object: Išmokytas modelis
        """
        if model_name not in self.models:
            raise ValueError(f"Modelis '{model_name}' nerastas!")
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"MOKYMASIS: {model_name}")
            print('='*60)
            print(f"Mokymo duomenys: {X_train.shape}")
        
        model = self.models[model_name]
        
        # Matuojame mokymo laiką
        start_time = time.time()
        
        model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        self.training_times[model_name] = training_time
        
        if verbose:
            print(f"✓ Mokymasis baigtas per {training_time:.2f} sekundes")
        
        self.trained_models[model_name] = model
        
        return model
    
    def train_all_models(self, X_train, y_train):
        """
        Mokosi visus modelius.
        
        Args:
            X_train (array): Mokymo požymiai
            y_train (array): Mokymo etiketės
            
        Returns:
            dict: Išmokyti modeliai
        """
        print("\n" + "="*60)
        print("VISŲ MODELIŲ MOKYMASIS")
        print("="*60)
        
        for model_name in self.models.keys():
            try:
                self.train_model(model_name, X_train, y_train)
            except Exception as e:
                print(f"✗ Klaida mokant {model_name}: {e}")
        
        print("\n" + "="*60)
        print("MOKYMASIS BAIGTAS")
        print("="*60)
        
        # Mokymo laikų suvestinė
        print("\nMokymo laikų palyginimas:")
        for name, time_val in sorted(self.training_times.items(), 
                                     key=lambda x: x[1]):
            print(f"  {name:20s}: {time_val:6.2f}s")
        
        return self.trained_models
    
    def cross_validate_model(self, model_name, X, y, cv=5):
        """
        Atlieka kryžminį validavimą.
        
        Args:
            model_name (str): Modelio pavadinimas
            X (array): Požymiai
            y (array): Etiketės
            cv (int): Fold'ų skaičius
            
        Returns:
            dict: Validavimo rezultatai
        """
        if model_name not in self.models:
            raise ValueError(f"Modelis '{model_name}' nerastas!")
        
        print(f"\nKryžminis validavimas: {model_name} (cv={cv})")
        
        model = self.models[model_name]
        
        # Skaičiuojame tikslumo įverčius
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        
        results = {
            'mean_accuracy': scores.mean(),
            'std_accuracy': scores.std(),
            'all_scores': scores
        }
        
        print(f"  Vidutinis tikslumas: {results['mean_accuracy']:.4f} "
              f"(± {results['std_accuracy']:.4f})")
        
        return results
    
    def get_feature_importance(self, model_name, feature_names):
        """
        Gauna požymių svarbą (tik Random Forest ir tree-based modeliams).
        
        Args:
            model_name (str): Modelio pavadinimas
            feature_names (list): Požymių pavadinimai
            
        Returns:
            pd.DataFrame: Požymių svarba
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Modelis '{model_name}' dar neišmokytas!")
        
        model = self.trained_models[model_name]
        
        # Tik Random Forest turi feature_importances_
        if not hasattr(model, 'feature_importances_'):
            print(f"⚠ {model_name} neturi feature importance")
            return None
        
        importance = model.feature_importances_
        
        # Sukuriame DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        return importance_df
    
    def save_model(self, model_name, filepath):
        """
        Išsaugo išmokytą modelį į failą.
        
        Args:
            model_name (str): Modelio pavadinimas
            filepath (str): Kelias kur saugoti
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Modelis '{model_name}' dar neišmokytas!")
        
        model = self.trained_models[model_name]
        joblib.dump(model, filepath)
        
        print(f"✓ Modelis '{model_name}' išsaugotas: {filepath}")
    
    def load_model(self, filepath, model_name=None):
        """
        Įkelia išmokytą modelį iš failo.
        
        Args:
            filepath (str): Kelias iki modelio failo
            model_name (str): Modelio pavadinimas (optional)
            
        Returns:
            object: Įkeltas modelis
        """
        model = joblib.load(filepath)
        
        if model_name:
            self.trained_models[model_name] = model
            print(f"✓ Modelis '{model_name}' įkeltas iš: {filepath}")
        else:
            print(f"✓ Modelis įkeltas iš: {filepath}")
        
        return model
    
    def predict(self, model_name, X):
        """
        Atlieka prognozavimą.
        
        Args:
            model_name (str): Modelio pavadinimas
            X (array): Požymiai
            
        Returns:
            array: Prognozės
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Modelis '{model_name}' dar neišmokytas!")
        
        model = self.trained_models[model_name]
        predictions = model.predict(X)
        
        return predictions
    
    def predict_proba(self, model_name, X):
        """
        Atlieka prognozavimą su tikimybėmis.
        
        Args:
            model_name (str): Modelio pavadinimas
            X (array): Požymiai
            
        Returns:
            array: Tikimybės
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Modelis '{model_name}' dar neišmokytas!")
        
        model = self.trained_models[model_name]
        
        if not hasattr(model, 'predict_proba'):
            raise ValueError(f"Modelis '{model_name}' neturi predict_proba")
        
        probabilities = model.predict_proba(X)
        
        return probabilities


def main():
    """Demonstracinis pavyzdys."""
    
    print("="*60)
    print("MODELIO MOKYMO DEMONSTRACIJA")
    print("="*60)
    
    # Generuojame sintetinius duomenis
    from sklearn.datasets import make_classification
    
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        weights=[0.7, 0.3],
        random_state=42
    )
    
    # Dalijame duomenis
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nDuomenys sugeneruoti:")
    print(f"  Mokymo: {X_train.shape}")
    print(f"  Testavimo: {X_test.shape}")
    
    # Inicializuojame trainerį
    trainer = ModelTrainer()
    trainer.initialize_models()
    
    # Mokome visus modelius
    trained_models = trainer.train_all_models(X_train, y_train)
    
    # Kryžminis validavimas Random Forest
    trainer.cross_validate_model('Random Forest', X_train, y_train, cv=5)
    
    # Požymių svarba
    feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
    importance_df = trainer.get_feature_importance('Random Forest', feature_names)
    
    if importance_df is not None:
        print("\nTop 5 svarbiausi požymiai:")
        print(importance_df.head())
    
    # Prognozavimas
    y_pred = trainer.predict('Random Forest', X_test)
    
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nRandom Forest tikslumas testavimo aibėje: {accuracy:.4f}")
    
    # Išsaugome modelį
    import os
    os.makedirs('/home/claude/iot_intrusion_detection/models', exist_ok=True)
    trainer.save_model('Random Forest', 
                      '/home/claude/iot_intrusion_detection/models/demo_rf.pkl')


if __name__ == "__main__":
    main()
