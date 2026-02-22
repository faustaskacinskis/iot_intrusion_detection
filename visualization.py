"""
Vizualizacijos modulis IoT atakų aptikimo sistemai.

Šis modulis kuria:
- Confusion Matrix heatmap
- ROC kreives
- Požymių svarbos grafikus
- Modelių palyginimo grafikus
- Mokymo istorijos grafikus
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# Nustatome stilių
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class Visualizer:
    """Klasė rezultatų vizualizavimui."""
    
    def __init__(self, figsize=(10, 6)):
        """
        Inicializuoja vizualizatorių.
        
        Args:
            figsize (tuple): Numatytasis grafikų dydis
        """
        self.figsize = figsize
        self.figures = {}
    
    def plot_confusion_matrix(self, cm, labels=None, title="Confusion Matrix", 
                            save_path=None):
        """
        Vizualizuoja confusion matrix.
        
        Args:
            cm (array): Confusion matrix
            labels (list): Klasių pavadinimai
            title (str): Grafiko pavadinimas
            save_path (str): Kelias išsaugojimui
        """
        if labels is None:
            labels = ['Normal', 'Attack']
        
        plt.figure(figsize=self.figsize)
        
        # Normalizuojame procentais
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Kuriame heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels,
                   cbar_kws={'label': 'Pavyzdžių skaičius'})
        
        # Pridedame procentus
        for i in range(len(labels)):
            for j in range(len(labels)):
                plt.text(j + 0.5, i + 0.7, f'({cm_percent[i, j]:.1f}%)',
                        ha='center', va='center', fontsize=9, color='gray')
        
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
        plt.ylabel('Tikroji klasė', fontsize=12)
        plt.xlabel('Prognozuota klasė', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Grafikas išsaugotas: {save_path}")
        
        self.figures['confusion_matrix'] = plt.gcf()
        plt.show()
    
    def plot_roc_curve(self, fpr, tpr, auc_score, model_name="Model", 
                      save_path=None):
        """
        Vizualizuoja ROC kreivę.
        
        Args:
            fpr (array): False positive rates
            tpr (array): True positive rates
            auc_score (float): AUC reikšmė
            model_name (str): Modelio pavadinimas
            save_path (str): Kelias išsaugojimui
        """
        plt.figure(figsize=self.figsize)
        
        # ROC kreivė
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'{model_name} (AUC = {auc_score:.4f})')
        
        # Diagonalė (atsitiktinis klasifikatorius)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                label='Atsitiktinis klasifikatorius (AUC = 0.5000)')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Kreivė - {model_name}', fontsize=14, fontweight='bold', pad=20)
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Grafikas išsaugotas: {save_path}")
        
        self.figures['roc_curve'] = plt.gcf()
        plt.show()
    
    def plot_multiple_roc_curves(self, roc_data, save_path=None):
        """
        Vizualizuoja kelių modelių ROC kreives.
        
        Args:
            roc_data (dict): {model_name: (fpr, tpr, auc_score)}
            save_path (str): Kelias išsaugojimui
        """
        plt.figure(figsize=self.figsize)
        
        colors = ['darkorange', 'green', 'red', 'purple', 'brown']
        
        for i, (model_name, (fpr, tpr, auc_score)) in enumerate(roc_data.items()):
            color = colors[i % len(colors)]
            plt.plot(fpr, tpr, color=color, lw=2,
                    label=f'{model_name} (AUC = {auc_score:.4f})')
        
        # Diagonalė
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                label='Atsitiktinis (AUC = 0.5000)')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Modelių ROC Kreivių Palyginimas', fontsize=14, 
                 fontweight='bold', pad=20)
        plt.legend(loc="lower right", fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Grafikas išsaugotas: {save_path}")
        
        self.figures['multiple_roc'] = plt.gcf()
        plt.show()
    
    def plot_feature_importance(self, importance_df, top_n=15, save_path=None):
        """
        Vizualizuoja požymių svarbą.
        
        Args:
            importance_df (pd.DataFrame): DataFrame su 'Feature' ir 'Importance'
            top_n (int): Kiek svarbiausių požymių rodyti
            save_path (str): Kelias išsaugojimui
        """
        plt.figure(figsize=(12, 8))
        
        # Imame top N požymių
        top_features = importance_df.head(top_n)
        
        # Horizontalus bar chart
        plt.barh(range(len(top_features)), top_features['Importance'], 
                color='steelblue', alpha=0.8)
        
        plt.yticks(range(len(top_features)), top_features['Feature'])
        plt.xlabel('Svarba (Importance)', fontsize=12)
        plt.ylabel('Požymis', fontsize=12)
        plt.title(f'Top {top_n} Svarbiausi Požymiai', fontsize=14, 
                 fontweight='bold', pad=20)
        plt.gca().invert_yaxis()  # Svarbiausias viršuje
        
        # Pridedame reikšmes
        for i, (idx, row) in enumerate(top_features.iterrows()):
            plt.text(row['Importance'], i, f" {row['Importance']:.4f}",
                    va='center', fontsize=9)
        
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Grafikas išsaugotas: {save_path}")
        
        self.figures['feature_importance'] = plt.gcf()
        plt.show()
    
    def plot_model_comparison(self, results_df, save_path=None):
        """
        Vizualizuoja kelių modelių palyginimą.
        
        Args:
            results_df (pd.DataFrame): Rezultatų DataFrame
            save_path (str): Kelias išsaugojimui
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Modelių Palyginimas', fontsize=16, fontweight='bold')
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            ax = axes[idx // 2, idx % 2]
            
            # Bar chart
            bars = ax.bar(results_df['model_name'], results_df[metric], 
                          color='steelblue', alpha=0.7)
            
            # Spalviname geriausią modelį
            best_idx = results_df[metric].idxmax()
            bars[best_idx].set_color('green')
            bars[best_idx].set_alpha(0.9)
            
            ax.set_ylabel(metric_name, fontsize=11)
            ax.set_title(metric_name, fontsize=12, fontweight='bold')
            ax.set_ylim([results_df[metric].min() * 0.95, 1.0])
            ax.grid(axis='y', alpha=0.3)
            
            # Sukame x labels
            ax.set_xticklabels(results_df['model_name'], rotation=45, ha='right')
            
            # Pridedame reikšmes ant barų
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}',
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Grafikas išsaugotas: {save_path}")
        
        self.figures['model_comparison'] = plt.gcf()
        plt.show()
    
    def plot_class_distribution(self, y, labels=None, title="Klasių Pasiskirstymas",
                               save_path=None):
        """
        Vizualizuoja klasių pasiskirstymą.
        
        Args:
            y (array): Klasių etiketės
            labels (list): Klasių pavadinimai
            title (str): Grafiko pavadinimas
            save_path (str): Kelias išsaugojimui
        """
        plt.figure(figsize=self.figsize)
        
        # Suskaičiuojame klases
        unique, counts = np.unique(y, return_counts=True)
        
        if labels is None:
            labels = [f'Class {i}' for i in unique]
        
        # Pie chart
        colors = ['lightgreen', 'lightcoral'] if len(unique) == 2 else None
        plt.pie(counts, labels=labels, autopct='%1.1f%%',
               colors=colors, startangle=90, textprops={'fontsize': 11})
        
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Pridedame legendą su skaičiais
        legend_labels = [f'{label}: {count:,}' for label, count in zip(labels, counts)]
        plt.legend(legend_labels, loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Grafikas išsaugotas: {save_path}")
        
        self.figures['class_distribution'] = plt.gcf()
        plt.show()
    
    def plot_training_time_comparison(self, training_times, save_path=None):
        """
        Vizualizuoja modelių mokymo laikų palyginimą.
        
        Args:
            training_times (dict): {model_name: time_seconds}
            save_path (str): Kelias išsaugojimui
        """
        plt.figure(figsize=self.figsize)
        
        models = list(training_times.keys())
        times = list(training_times.values())
        
        # Bar chart
        bars = plt.bar(models, times, color='steelblue', alpha=0.7)
        
        # Spalviname greičiausią
        min_idx = times.index(min(times))
        bars[min_idx].set_color('green')
        bars[min_idx].set_alpha(0.9)
        
        plt.ylabel('Laikas (sekundės)', fontsize=12)
        plt.title('Modelių Mokymo Laikų Palyginimas', fontsize=14, 
                 fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Pridedame reikšmes
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}s',
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Grafikas išsaugotas: {save_path}")
        
        self.figures['training_times'] = plt.gcf()
        plt.show()
    
    def plot_precision_recall_curve(self, y_true, y_proba, save_path=None):
        """
        Vizualizuoja Precision-Recall kreivę.
        
        Args:
            y_true (array): Tikrosios etiketės
            y_proba (array): Prognozuotos tikimybės
            save_path (str): Kelias išsaugojimui
        """
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        # Jei y_proba yra 2D
        if len(y_proba.shape) > 1:
            y_proba = y_proba[:, 1]
        
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        ap_score = average_precision_score(y_true, y_proba)
        
        plt.figure(figsize=self.figsize)
        
        plt.plot(recall, precision, color='darkorange', lw=2,
                label=f'Precision-Recall (AP = {ap_score:.4f})')
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Kreivė', fontsize=14, fontweight='bold', pad=20)
        plt.legend(loc="lower left", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Grafikas išsaugotas: {save_path}")
        
        self.figures['pr_curve'] = plt.gcf()
        plt.show()
    
    def create_results_dashboard(self, cm, results_df, importance_df, 
                                 roc_data, save_path=None):
        """
        Sukuria bendrą rezultatų dashboard su visais grafikais.
        
        Args:
            cm (array): Confusion matrix
            results_df (pd.DataFrame): Modelių rezultatai
            importance_df (pd.DataFrame): Požymių svarba
            roc_data (dict): ROC duomenys
            save_path (str): Kelias išsaugojimui
        """
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Confusion Matrix
        ax1 = fig.add_subplot(gs[0, 0])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=['Normal', 'Attack'],
                   yticklabels=['Normal', 'Attack'])
        ax1.set_title('Confusion Matrix', fontweight='bold')
        
        # 2. Modelių palyginimas - F1 scores
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.bar(results_df['model_name'], results_df['f1_score'], 
               color='steelblue', alpha=0.7)
        ax2.set_title('F1-Score Palyginimas', fontweight='bold')
        ax2.set_xticklabels(results_df['model_name'], rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. Požymių svarba (top 10)
        ax3 = fig.add_subplot(gs[0, 2])
        top10 = importance_df.head(10)
        ax3.barh(range(len(top10)), top10['Importance'], color='steelblue', alpha=0.7)
        ax3.set_yticks(range(len(top10)))
        ax3.set_yticklabels(top10['Feature'], fontsize=8)
        ax3.set_title('Top 10 Požymių', fontweight='bold')
        ax3.invert_yaxis()
        ax3.grid(axis='x', alpha=0.3)
        
        # 4-6. ROC kreivės (po vieną kiekvienam modeliui)
        for idx, (model_name, (fpr, tpr, auc_score)) in enumerate(list(roc_data.items())[:3]):
            ax = fig.add_subplot(gs[1, idx])
            ax.plot(fpr, tpr, color='darkorange', lw=2,
                   label=f'AUC = {auc_score:.4f}')
            ax.plot([0, 1], [0, 1], 'k--', lw=1)
            ax.set_title(f'ROC: {model_name}', fontweight='bold', fontsize=10)
            ax.legend(loc="lower right", fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # 7-9. Metrikos pagal modelį
        metrics_to_plot = ['accuracy', 'precision', 'recall']
        for idx, metric in enumerate(metrics_to_plot):
            ax = fig.add_subplot(gs[2, idx])
            ax.bar(results_df['model_name'], results_df[metric],
                  color='steelblue', alpha=0.7)
            ax.set_title(metric.capitalize(), fontweight='bold')
            ax.set_xticklabels(results_df['model_name'], rotation=45, ha='right', fontsize=8)
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim([0.9, 1.0])
        
        fig.suptitle('IoT Atakų Aptikimo Sistemos Rezultatų Dashboard', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Dashboard išsaugotas: {save_path}")
        
        self.figures['dashboard'] = fig
        plt.show()


def main():
    """Demonstracinis pavyzdys."""
    
    print("="*60)
    print("VIZUALIZACIJOS DEMONSTRACIJA")
    print("="*60)
    
    # Inicializuojame vizualizatorių
    viz = Visualizer()
    
    # Demo confusion matrix
    cm = np.array([[850, 30], [20, 100]])
    viz.plot_confusion_matrix(cm, labels=['Normal', 'Attack'],
                             title="Demo Confusion Matrix")
    
    # Demo ROC curve
    fpr = np.linspace(0, 1, 100)
    tpr = np.sqrt(fpr)  # Demo curve
    viz.plot_roc_curve(fpr, tpr, 0.95, model_name="Demo Model")
    
    # Demo feature importance
    importance_data = {
        'Feature': [f'feature_{i}' for i in range(20)],
        'Importance': np.random.exponential(0.1, 20)
    }
    importance_df = pd.DataFrame(importance_data).sort_values('Importance', ascending=False)
    viz.plot_feature_importance(importance_df, top_n=10)
    
    # Demo model comparison
    results_data = {
        'model_name': ['Random Forest', 'SVM', 'KNN', 'Neural Net'],
        'accuracy': [0.992, 0.985, 0.978, 0.990],
        'precision': [0.991, 0.983, 0.975, 0.989],
        'recall': [0.993, 0.987, 0.980, 0.991],
        'f1_score': [0.992, 0.985, 0.977, 0.990]
    }
    results_df = pd.DataFrame(results_data)
    viz.plot_model_comparison(results_df)
    
    print("\n✓ Vizualizacijos demonstracija baigta")


if __name__ == "__main__":
    main()
