"""
Clasificador PAC basado en Firmas de Islas.

Experimento: en lugar de clasificar MNIST usando píxeles brutos (784D),
usamos firmas de islas (56D = 28 horizontales + 28 verticales).

Esto reduce drásticamente la dimensionalidad y podría capturar
la estructura morfológica esencial de cada dígito.

Se compara:
  - PAC-V2 con firmas de islas (56D)
  - PAC-V2 con píxeles brutos (784D) — baseline
"""

import os
import sys
import time
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# Añadir directorio padre al path para importar pac_v2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from pac_v2 import PurifyingArchetypeClassifierV2


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_DIR = "../results/island_classifier"
os.makedirs(RESULTS_DIR, exist_ok=True)


# =============================================================================
# FUNCIONES
# =============================================================================

def load_mnist():
    """Carga MNIST como tensores (N, 1, 28, 28) y etiquetas."""
    print("Loading MNIST dataset...")
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('../data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=60000, shuffle=False)
    x_train, y_train = next(iter(train_loader))

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10000, shuffle=False)
    x_test, y_test = next(iter(test_loader))

    print(f"Train: {x_train.shape[0]}, Test: {x_test.shape[0]}")
    return x_train, y_train, x_test, y_test


def compute_island_signatures(images):
    """
    Calcula firmas de islas (56D) para un lote de imágenes.

    Returns
    -------
    signatures : torch.Tensor, shape (N, 56)
        28 valores horizontales + 28 valores verticales.
    """
    N = images.shape[0]
    binary = (images.squeeze(1) > 0.0).float()  # (N, 28, 28)

    # --- Firmas horizontales (islas por fila) ---
    H, W = 28, 28
    rows = binary.view(N * H, W)  # (N*28, 28)
    diff_h = rows[:, 1:] - rows[:, :-1]  # (N*28, 27)
    h_counts = (diff_h == 1).sum(dim=1) + rows[:, 0]  # (N*28,)
    h_sigs = h_counts.view(N, H)  # (N, 28)

    # --- Firmas verticales (islas por columna) ---
    cols = binary.transpose(1, 2).reshape(N * W, H)  # (N*28, 28)
    diff_v = cols[:, 1:] - cols[:, :-1]
    v_counts = (diff_v == 1).sum(dim=1) + cols[:, 0]
    v_sigs = v_counts.view(N, W)  # (N, 28)

    # Concatenar
    signatures = torch.cat([h_sigs, v_sigs], dim=1).float()  # (N, 56)
    return signatures


def compute_confusion_matrix(y_true, y_pred, num_classes=10):
    """Calcula matriz de confusión usando PyTorch/NumPy."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def train_and_evaluate(name, x_train, y_train, x_test, y_test, max_iters=100):
    """
    Entrena PAC-V2 y evalúa. Devuelve métricas.
    """
    print(f"\n{'='*60}")
    print(f"Entrenando PAC-V2: {name}")
    print(f"Dimensiones: {x_train.shape[1]}D")
    print(f"{'='*60}")

    pac = PurifyingArchetypeClassifierV2(max_iters=max_iters, target_acc=0.999)

    start_time = time.time()
    pac.fit(x_train, y_train, verbose=True)
    train_time = time.time() - start_time

    # Evaluar en test
    preds, confidences = pac.predict(x_test)
    preds_np = preds.cpu().numpy()
    y_test_np = y_test.cpu().numpy()

    correct = (preds_np == y_test_np)
    accuracy = correct.mean()
    num_archetypes = len(pac.arch_labels)

    # Matriz de confusión
    cm = compute_confusion_matrix(y_test_np, preds_np)

    print(f"\n--- Resultados {name} ---")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Archetypes descubiertos: {num_archetypes}")
    print(f"Tiempo de entrenamiento: {train_time:.2f}s")

    return {
        'name': name,
        'accuracy': float(accuracy),
        'num_archetypes': int(num_archetypes),
        'train_time': float(train_time),
        'predictions': preds_np,
        'confusion_matrix': cm,
        'pac': pac
    }


def plot_confusion_matrices(results_sig, results_pix, save_path):
    """Visualiza matrices de confusión lado a lado."""
    print(f"\n  Generando confusion matrices -> {os.path.basename(save_path)}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, results, title in zip(axes,
                                   [results_sig, results_pix],
                                   ['Firmas de Islas (56D)', 'Píxeles Brutos (784D)']):
        cm = results['confusion_matrix']
        # Normalizar por fila (true label)
        cm_norm = cm.astype(np.float64)
        row_sums = cm_norm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = cm_norm / row_sums

        im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
        ax.set_title(f'{title}\\nAcc: {results["accuracy"]*100:.2f}%', fontsize=13)
        ax.set_xlabel('Predicho', fontsize=11)
        ax.set_ylabel('Real', fontsize=11)
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))

        # Añadir valores
        for i in range(10):
            for j in range(10):
                val = cm[i, j]
                if val > 0:
                    text_color = 'white' if cm_norm[i, j] > 0.5 else 'black'
                    ax.text(j, i, str(val), ha='center', va='center',
                           fontsize=7, color=text_color)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_comparison_bar(results_sig, results_pix, save_path):
    """Compara métricas clave en barras."""
    print(f"  Generando comparison chart -> {os.path.basename(save_path)}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = ['Accuracy (%)', 'Archetypes', 'Train Time (s)']
    sig_values = [
        results_sig['accuracy'] * 100,
        results_sig['num_archetypes'],
        results_sig['train_time']
    ]
    pix_values = [
        results_pix['accuracy'] * 100,
        results_pix['num_archetypes'],
        results_pix['train_time']
    ]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = axes[0].bar(x, sig_values, width, label='Firmas (56D)', color='steelblue')
    bars2 = axes[0].bar(x + width, pix_values, width, label='Píxeles (784D)', color='coral')

    axes[0].set_ylabel('Valor')
    axes[0].set_title('Comparación PAC-V2: Firmas vs Píxeles')
    axes[0].set_xticks(x + width / 2)
    axes[0].set_xticklabels(metrics)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # Añadir valores sobre barras
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[0].annotate(f'{height:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)

    # Accuracy zoom
    axes[1].bar(['Firmas (56D)', 'Píxeles (784D)'],
               [results_sig['accuracy']*100, results_pix['accuracy']*100],
               color=['steelblue', 'coral'])
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Accuracy en Test Set')
    axes[1].set_ylim([0, 100])
    axes[1].grid(True, alpha=0.3, axis='y')

    # Archetypes comparison
    axes[2].bar(['Firmas (56D)', 'Píxeles (784D)'],
               [results_sig['num_archetypes'], results_pix['num_archetypes']],
               color=['steelblue', 'coral'])
    axes[2].set_ylabel('Número de Archetypes')
    axes[2].set_title('Archetypes Descubiertos')
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_signature_archetypes(results_sig, save_path):
    """
    Visualiza los archetypes descubiertos en el espacio de firmas.
    Muestra las firmas de los archetypes como heatmaps.
    """
    print(f"  Generando archetype signatures -> {os.path.basename(save_path)}")

    pac = results_sig['pac']
    arch_tensors = pac.arch_tensors.cpu().numpy()  # (num_arch, 56)
    arch_labels = pac.arch_labels.cpu().numpy()

    # Ordenar por label
    sorted_idx = np.argsort(arch_labels)
    arch_tensors = arch_tensors[sorted_idx]
    arch_labels = arch_labels[sorted_idx]

    # Seleccionar hasta 50 archetypes para visualizar
    max_display = min(50, len(arch_labels))
    arch_tensors = arch_tensors[:max_display]
    arch_labels = arch_labels[:max_display]

    fig, axes = plt.subplots(1, 2, figsize=(16, 10))

    # Heatmap horizontal (28 cols = filas de la imagen)
    h_part = arch_tensors[:, :28]
    im1 = axes[0].imshow(h_part, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    axes[0].set_title('Archetypes: Firmas Horizontales (islas/fila)', fontsize=12)
    axes[0].set_xlabel('Fila de imagen')
    axes[0].set_ylabel('Archetype (ordenado por dígito)')
    for i in range(len(arch_labels)):
        axes[0].text(-2, i, str(arch_labels[i]), ha='right', va='center', fontsize=8)
    plt.colorbar(im1, ax=axes[0])

    # Heatmap vertical (28 cols = columnas de la imagen)
    v_part = arch_tensors[:, 28:]
    im2 = axes[1].imshow(v_part, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    axes[1].set_title('Archetypes: Firmas Verticales (islas/columna)', fontsize=12)
    axes[1].set_xlabel('Columna de imagen')
    axes[1].set_ylabel('Archetype (ordenado por dígito)')
    for i in range(len(arch_labels)):
        axes[1].text(-2, i, str(arch_labels[i]), ha='right', va='center', fontsize=8)
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("CLASIFICADOR PAC BASADO EN FIRMAS DE ISLAS")
    print("=" * 70)

    # 1. Cargar datos
    x_train, y_train, x_test, y_test = load_mnist()

    # 2. Preparar representaciones
    print("\nPreparando representaciones...")

    # Firmas de islas (56D)
    print("  - Computing island signatures...")
    sig_train = compute_island_signatures(x_train)
    sig_test = compute_island_signatures(x_test)

    # Píxeles brutos (784D)
    pix_train = x_train.view(60000, 784)
    pix_test = x_test.view(10000, 784)

    print(f"  - Signatures shape: {sig_train.shape}")
    print(f"  - Pixels shape: {pix_train.shape}")

    # 3. Entrenar y evaluar con firmas
    results_sig = train_and_evaluate(
        "Island Signatures (56D)",
        sig_train, y_train, sig_test, y_test,
        max_iters=100
    )

    # 4. Entrenar y evaluar con píxeles (baseline)
    results_pix = train_and_evaluate(
        "Raw Pixels (784D)",
        pix_train, y_train, pix_test, y_test,
        max_iters=100
    )

    # 5. Guardar resultados
    results_summary = {
        'island_signatures': {
            'accuracy': results_sig['accuracy'],
            'num_archetypes': results_sig['num_archetypes'],
            'train_time': results_sig['train_time'],
        },
        'raw_pixels': {
            'accuracy': results_pix['accuracy'],
            'num_archetypes': results_pix['num_archetypes'],
            'train_time': results_pix['train_time'],
        }
    }

    json_path = os.path.join(RESULTS_DIR, "classifier_comparison.json")
    with open(json_path, 'w') as f:
        json.dump(results_summary, f, indent=2)

    # 6. Visualizaciones
    print("\nGenerando visualizaciones...")
    plot_confusion_matrices(results_sig, results_pix,
                           os.path.join(RESULTS_DIR, "confusion_matrices.png"))
    plot_comparison_bar(results_sig, results_pix,
                       os.path.join(RESULTS_DIR, "comparison_metrics.png"))
    plot_signature_archetypes(results_sig,
                             os.path.join(RESULTS_DIR, "signature_archetypes.png"))

    # 7. Resumen final
    print("\n" + "=" * 70)
    print("RESUMEN FINAL")
    print("=" * 70)
    print(f"{'Métrica':<25} {'Firmas (56D)':>15} {'Píxeles (784D)':>15}")
    print("-" * 70)
    print(f"{'Accuracy':<25} {results_sig['accuracy']*100:>14.2f}% {results_pix['accuracy']*100:>14.2f}%")
    print(f"{'Archetypes':<25} {results_sig['num_archetypes']:>15} {results_pix['num_archetypes']:>15}")
    print(f"{'Train Time (s)':<25} {results_sig['train_time']:>15.2f} {results_pix['train_time']:>15.2f}")
    print(f"{'Dimensión':<25} {'56':>15} {'784':>15}")
    print("-" * 70)

    speedup = results_pix['train_time'] / results_sig['train_time'] if results_sig['train_time'] > 0 else 0
    print(f"\\nSpeedup en entrenamiento: {speedup:.1f}x más rápido con firmas")
    print(f"Reducción de dimensionalidad: {784/56:.1f}x")

    print(f"\\nResultados guardados en: {RESULTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
