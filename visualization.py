"""
=============================================================================
Modulis: visualization.py
Aprašymas: Visų grafikų generavimas – confusion matrix, ROC, požymių svarba
=============================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Globalus stilius
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor':   '#161b22',
    'axes.edgecolor':   '#30363d',
    'axes.labelcolor':  '#e6edf3',
    'xtick.color':      '#8b949e',
    'ytick.color':      '#8b949e',
    'text.color':       '#e6edf3',
    'grid.color':       '#21262d',
    'grid.alpha':        0.5,
    'font.family':      'monospace',
})

PALETTE = {
    'green':  '#3fb950',
    'blue':   '#58a6ff',
    'orange': '#f0883e',
    'red':    '#f85149',
    'purple': '#bc8cff',
    'teal':   '#39d353',
}

OUTPUT_DIR = 'results'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================================
# 1. Klasių pasiskirstymas
# =========================================================================

def plot_class_distribution(df, category_col='category', save=True):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('BoT-IoT: Duomenų pasiskirstymas', fontsize=14,
                 color='#e6edf3', fontweight='bold')

    # Pyrago diagrama
    counts = df[category_col].value_counts()
    colors = [PALETTE['green'], PALETTE['red'], PALETTE['orange'],
              PALETTE['blue'], PALETTE['purple']][:len(counts)]
    axes[0].pie(counts.values, labels=counts.index, autopct='%1.1f%%',
                colors=colors, textprops={'color': '#e6edf3', 'fontsize': 9},
                wedgeprops={'edgecolor': '#0d1117', 'linewidth': 1.5})
    axes[0].set_title('Atakų kategorijų dalis', color='#e6edf3')

    # Stulpelių diagrama
    bars = axes[1].bar(counts.index, counts.values, color=colors,
                       edgecolor='#0d1117', linewidth=1)
    for bar, val in zip(bars, counts.values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                     f'{val:,}', ha='center', va='bottom',
                     color='#e6edf3', fontsize=8)
    axes[1].set_title('Įrašų skaičius pagal kategoriją', color='#e6edf3')
    axes[1].set_xlabel('Kategorija')
    axes[1].set_ylabel('Skaičius')
    plt.xticks(rotation=20)

    plt.tight_layout()
    if save:
        path = f'{OUTPUT_DIR}/01_class_distribution.png'
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
        print(f"[IŠSAUGOTA] {path}")
    plt.show()


# =========================================================================
# 2. Palyginamoji modelių metrikų diagrama
# =========================================================================

def plot_model_comparison(results: dict, save=True):
    model_names = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    labels  = ['Accuracy', 'Precision', 'Recall', 'F1-score']
    bar_colors = [PALETTE['blue'], PALETTE['green'], PALETTE['orange'], PALETTE['purple']]

    x = np.arange(len(model_names))
    width = 0.18

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor('#0d1117')

    for i, (metric, label, color) in enumerate(zip(metrics, labels, bar_colors)):
        vals = [results[m][metric] for m in model_names]
        bars = ax.bar(x + i * width, vals, width,
                      label=label, color=color, alpha=0.85,
                      edgecolor='#0d1117', linewidth=0.8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'{v:.3f}', ha='center', va='bottom',
                    color='#e6edf3', fontsize=7)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names, fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel('Reikšmė')
    ax.set_title('Modelių palyginimas – pagrindinės metrikos',
                 fontsize=13, fontweight='bold', color='#e6edf3', pad=12)
    ax.legend(loc='lower right', framealpha=0.3)
    ax.yaxis.grid(True, alpha=0.4)

    plt.tight_layout()
    if save:
        path = f'{OUTPUT_DIR}/02_model_comparison.png'
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
        print(f"[IŠSAUGOTA] {path}")
    plt.show()


# =========================================================================
# 3. Confusion Matrix kiekvienam modeliui
# =========================================================================

def plot_confusion_matrices(results: dict, save=True):
    n = len(results)
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('Sumaišties matricos (Confusion Matrices)',
                 fontsize=14, fontweight='bold', color='#e6edf3')
    axes_flat = axes.flatten()

    cmap = sns.color_palette(
        ["#0d1117", "#0e4429", "#006d32", "#26a641", "#39d353"], as_cmap=True
    )

    for ax, (name, res) in zip(axes_flat, results.items()):
        cm = res['confusion_matrix']
        class_names = res['class_names'] or [str(i) for i in range(cm.shape[0])]

        # Normalizuota matrica procentais
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        sns.heatmap(cm_norm, annot=True, fmt='.2%', ax=ax,
                    cmap='YlOrRd', linewidths=0.5,
                    linecolor='#0d1117',
                    xticklabels=class_names,
                    yticklabels=class_names,
                    cbar_kws={'shrink': 0.8})

        ax.set_title(f'{name}\nF1={res["f1"]:.4f}',
                     color='#e6edf3', fontsize=10, fontweight='bold')
        ax.set_xlabel('Prognozuota klasė', fontsize=9)
        ax.set_ylabel('Tikroji klasė', fontsize=9)
        ax.tick_params(colors='#8b949e', labelsize=8)
        plt.setp(ax.get_xticklabels(), rotation=20)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save:
        path = f'{OUTPUT_DIR}/03_confusion_matrices.png'
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
        print(f"[IŠSAUGOTA] {path}")
    plt.show()


# =========================================================================
# 4. ROC kreivės
# =========================================================================

def plot_roc_curves(results: dict, save=True):
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor('#0d1117')

    colors = list(PALETTE.values())
    has_roc = False

    for (name, res), color in zip(results.items(), colors):
        if res.get('roc_auc') is not None:
            ax.plot(res['fpr'], res['tpr'],
                    label=f'{name} (AUC={res["roc_auc"]:.4f})',
                    color=color, linewidth=2)
            has_roc = True

    if has_roc:
        ax.plot([0, 1], [0, 1], '--', color='#8b949e', linewidth=1, label='Atsitiktinis')
        ax.fill_between([0, 1], [0, 1], alpha=0.05, color='#8b949e')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.set_xlabel('False Positive Rate (FPR)', fontsize=11)
        ax.set_ylabel('True Positive Rate (TPR)', fontsize=11)
        ax.set_title('ROC kreivės – dvejetainė klasifikacija',
                     fontsize=13, fontweight='bold', color='#e6edf3', pad=12)
        ax.legend(loc='lower right', framealpha=0.3, fontsize=10)
        ax.yaxis.grid(True, alpha=0.4)
    else:
        ax.text(0.5, 0.5, 'ROC galima tik dvejetainei klasifikacijai',
                ha='center', va='center', color='#8b949e', fontsize=12)

    plt.tight_layout()
    if save:
        path = f'{OUTPUT_DIR}/04_roc_curves.png'
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
        print(f"[IŠSAUGOTA] {path}")
    plt.show()


# =========================================================================
# 5. Požymių svarba (Random Forest)
# =========================================================================

def plot_feature_importance(importance_dict: dict, top_n: int = 15, save=True):
    items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names, vals = zip(*items)

    # Spalvų gradientas pagal svarbą
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.95, len(names)))[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#0d1117')

    bars = ax.barh(list(names)[::-1], list(vals)[::-1],
                   color=colors, edgecolor='#0d1117', linewidth=0.5)
    for bar, v in zip(bars, list(vals)[::-1]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{v:.4f}', va='center', color='#e6edf3', fontsize=8)

    ax.set_xlabel('Požymio svarba (Gini impurity)', fontsize=11)
    ax.set_title(f'Top-{top_n} svarbiausi požymiai (Random Forest)',
                 fontsize=13, fontweight='bold', color='#e6edf3', pad=12)
    ax.xaxis.grid(True, alpha=0.4)
    plt.tight_layout()

    if save:
        path = f'{OUTPUT_DIR}/05_feature_importance.png'
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
        print(f"[IŠSAUGOTA] {path}")
    plt.show()


# =========================================================================
# 6. Mokymo laikų palyginimas
# =========================================================================

def plot_training_times(results: dict, save=True):
    names  = list(results.keys())
    times  = [results[n]['train_time'] for n in names]
    infer  = [results[n]['inference_time'] * 1000 for n in names]  # ms

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('#0d1117')

    bar_colors = [PALETTE['blue'], PALETTE['orange'], PALETTE['green'], PALETTE['purple']]

    # Mokymo laikas
    bars1 = ax1.bar(names, times, color=bar_colors, edgecolor='#0d1117')
    for b, t in zip(bars1, times):
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 0.05,
                 f'{t:.1f}s', ha='center', color='#e6edf3', fontsize=9)
    ax1.set_title('Mokymo laikas', color='#e6edf3', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Sekundės')
    ax1.yaxis.grid(True, alpha=0.4)

    # Prognozavimo laikas (ms)
    bars2 = ax2.bar(names, infer, color=bar_colors, edgecolor='#0d1117')
    for b, t in zip(bars2, infer):
        ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.1,
                 f'{t:.1f}ms', ha='center', color='#e6edf3', fontsize=9)
    ax2.set_title('Prognozavimo laikas (visos testavimo aibės)',
                  color='#e6edf3', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Milisekundės')
    ax2.yaxis.grid(True, alpha=0.4)

    plt.suptitle('Modelių našumas – laikas', fontsize=13, fontweight='bold',
                 color='#e6edf3', y=1.02)
    plt.tight_layout()

    if save:
        path = f'{OUTPUT_DIR}/06_training_times.png'
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
        print(f"[IŠSAUGOTA] {path}")
    plt.show()
