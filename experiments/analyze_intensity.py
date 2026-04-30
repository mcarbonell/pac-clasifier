"""
Análisis de intensidad de píxeles en MNIST.

Este script calcula, para cada imagen del dataset MNIST, la suma total de
intensidades de todos los píxeles. Esto equivale a medir la "cantidad de tinta"
o grosor del trazo de cada dígito.

El objetivo es cuantificar la variabilidad en la intensidad total y detectar
diferencias sistemáticas entre clases (ej: '1' es más ligero que '8').
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_DIR = "../results/intensity_analysis"

os.makedirs(RESULTS_DIR, exist_ok=True)


# =============================================================================
# FUNCIONES DE CÁLCULO
# =============================================================================

def load_mnist():
    """Carga MNIST (train + test) como tensores (N, 1, 28, 28) e índices."""
    print("Loading MNIST dataset...")
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('../data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=60000, shuffle=False)
    x_train, y_train = next(iter(train_loader))

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10000, shuffle=False)
    x_test, y_test = next(iter(test_loader))

    x_all = torch.cat([x_train, x_test], dim=0)
    y_all = torch.cat([y_train, y_test], dim=0)

    print(f"Total images loaded: {x_all.shape[0]}")
    return x_all, y_all


def compute_total_intensity(images):
    """
    Calcula la intensidad total por imagen: suma de todos los píxeles.

    Como MNIST con ToTensor() está en [0, 1], el resultado está en [0, 784].
    """
    return images.sum(dim=(1, 2, 3))  # (N,)


def compute_statistics(intensities, labels):
    """Calcula estadísticas globales y por dígito."""
    intens_np = intensities.cpu().numpy()
    labels_np = labels.cpu().numpy()

    stats = {
        'global': {
            'count': int(len(intens_np)),
            'mean': float(np.mean(intens_np)),
            'std': float(np.std(intens_np)),
            'min': float(np.min(intens_np)),
            'max': float(np.max(intens_np)),
            'p5': float(np.percentile(intens_np, 5)),
            'p25': float(np.percentile(intens_np, 25)),
            'p50': float(np.percentile(intens_np, 50)),
            'p75': float(np.percentile(intens_np, 75)),
            'p95': float(np.percentile(intens_np, 95)),
        },
        'by_digit': {}
    }

    for digit in range(10):
        mask = labels_np == digit
        vals = intens_np[mask]

        stats['by_digit'][str(digit)] = {
            'count': int(mask.sum()),
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals)),
            'min': float(np.min(vals)),
            'max': float(np.max(vals)),
            'p5': float(np.percentile(vals, 5)),
            'p50': float(np.percentile(vals, 50)),
            'p95': float(np.percentile(vals, 95)),
        }

    return stats


# =============================================================================
# VISUALIZACIONES
# =============================================================================

def plot_intensity_histogram(intensities, labels, save_path):
    """Histograma global de intensidad total con distribución por dígito."""
    print(f"  Generating histogram -> {os.path.basename(save_path)}")
    intens_np = intensities.cpu().numpy()
    labels_np = labels.cpu().numpy()

    fig, ax = plt.subplots(figsize=(12, 6))

    # Histograma global
    ax.hist(intens_np, bins=100, color='lightgray', edgecolor='black', alpha=0.5, label='Global', density=True)

    # Histogramas por dígito (sobrepuestos)
    for digit in range(10):
        mask = labels_np == digit
        ax.hist(intens_np[mask], bins=60, alpha=0.4, label=str(digit),
                color=plt.cm.tab10(digit / 9.0), density=True, histtype='step', linewidth=1.5)

    ax.set_xlabel('Intensidad Total (suma de píxeles)', fontsize=12)
    ax.set_ylabel('Densidad', fontsize=12)
    ax.set_title('Distribución de Intensidad Total por Dígito', fontsize=14)
    ax.legend(title='Dígito', loc='upper right')
    ax.set_xlim(0, 300)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_boxplots_by_digit(intensities, labels, save_path):
    """Boxplots de intensidad total agrupados por dígito."""
    print(f"  Generating boxplots -> {os.path.basename(save_path)}")
    intens_np = intensities.cpu().numpy()
    labels_np = labels.cpu().numpy()

    fig, ax = plt.subplots(figsize=(12, 6))

    data = [intens_np[labels_np == d] for d in range(10)]

    bp = ax.boxplot(data, tick_labels=range(10), patch_artist=True,
                    showfliers=False,
                    medianprops=dict(color='red', linewidth=2))

    for patch, color in zip(bp['boxes'], plt.cm.tab10(np.arange(10) / 9.0)):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_xlabel('Dígito', fontsize=12)
    ax.set_ylabel('Intensidad Total', fontsize=12)
    ax.set_title('Intensidad Total por Dígito (sin outliers)', fontsize=14)

    # Añadir líneas de referencia para media global
    global_mean = np.mean(intens_np)
    ax.axhline(global_mean, color='black', linestyle='--', alpha=0.5, label=f'Media global: {global_mean:.1f}')
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_density_by_digit(intensities, labels, save_path):
    """Distribución de densidad de intensidad para cada dígito."""
    print(f"  Generating density plot -> {os.path.basename(save_path)}")
    intens_np = intensities.cpu().numpy()
    labels_np = labels.cpu().numpy()

    fig, ax = plt.subplots(figsize=(12, 7))

    for digit in range(10):
        mask = labels_np == digit
        vals = intens_np[mask]

        # Histograma suavizado con muchos bins
        counts, bins = np.histogram(vals, bins=80, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Suavizado simple con convolución
        window = np.ones(3) / 3
        smoothed = np.convolve(counts, window, mode='same')

        ax.plot(bin_centers, smoothed, label=str(digit),
                color=plt.cm.tab10(digit / 9.0), linewidth=2, alpha=0.8)
        ax.fill_between(bin_centers, smoothed, alpha=0.1, color=plt.cm.tab10(digit / 9.0))

    ax.set_xlabel('Intensidad Total', fontsize=12)
    ax.set_ylabel('Densidad', fontsize=12)
    ax.set_title('Distribución de Intensidad por Dígito', fontsize=14)
    ax.legend(title='Dígito', loc='upper right')
    ax.set_xlim(0, 300)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_examples_by_intensity(images, intensities, labels, save_path):
    """
    Muestra ejemplos de cada dígito clasificados por intensidad:
    baja (percentil 10), media (percentil 50), alta (percentil 90).
    """
    print(f"  Generating examples grid -> {os.path.basename(save_path)}")
    labels_np = labels.cpu().numpy()
    intens_np = intensities.cpu().numpy()

    fig, axes = plt.subplots(10, 3, figsize=(9, 22))

    categories = ['Baja intensidad\n(P10)', 'Media intensidad\n(P50)', 'Alta intensidad\n(P90)']
    percentiles = [10, 50, 90]

    for digit in range(10):
        mask = labels_np == digit
        idxs = np.where(mask)[0]
        vals = intens_np[mask]

        for col, (cat, p) in enumerate(zip(categories, percentiles)):
            ax = axes[digit, col]
            target = np.percentile(vals, p)
            # Encontrar la imagen más cercana al percentil objetivo
            closest_idx = idxs[np.argmin(np.abs(vals - target))]
            img = images[closest_idx, 0].cpu().numpy()

            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            ax.axis('off')

            # Mostrar valor de intensidad
            actual_val = intens_np[closest_idx]
            ax.text(0.5, -0.05, f'{actual_val:.1f}', transform=ax.transAxes,
                    ha='center', va='top', fontsize=10, color='blue')

            if digit == 0:
                ax.set_title(cat, fontsize=11)

        # Etiqueta de fila
        axes[digit, 0].set_ylabel(str(digit), rotation=0, fontsize=16, labelpad=20, va='center')

    fig.suptitle('Ejemplos por Nivel de Intensidad\n(valor numérico = suma de píxeles)', fontsize=16, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_intensity_vs_translation(intensities, deviations, labels, save_path):
    """
    Scatter plot de intensidad vs desviación del centro de gravedad.
    Muestra si hay correlación entre cantidad de tinta y traslación.
    """
    print(f"  Generating intensity vs translation -> {os.path.basename(save_path)}")
    intens_np = intensities.cpu().numpy()
    dx = deviations[:, 0].cpu().numpy()
    dy = deviations[:, 1].cpu().numpy()
    labels_np = labels.cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Intensidad vs dx
    for digit in range(10):
        mask = labels_np == digit
        axes[0].scatter(dx[mask], intens_np[mask], s=1, alpha=0.3,
                        color=plt.cm.tab10(digit / 9.0), label=str(digit) if digit == 0 or digit == 5 else None)

    axes[0].set_xlabel('Desviación X (píxeles)', fontsize=11)
    axes[0].set_ylabel('Intensidad Total', fontsize=11)
    axes[0].set_title('Intensidad vs Desviación en X')
    axes[0].axvline(0, color='black', linestyle='--', alpha=0.3)

    # Intensidad vs dy
    for digit in range(10):
        mask = labels_np == digit
        axes[1].scatter(dy[mask], intens_np[mask], s=1, alpha=0.3,
                        color=plt.cm.tab10(digit / 9.0))

    axes[1].set_xlabel('Desviación Y (píxeles)', fontsize=11)
    axes[1].set_ylabel('Intensidad Total', fontsize=11)
    axes[1].set_title('Intensidad vs Desviación en Y')
    axes[1].axvline(0, color='black', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("ANÁLISIS DE INTENSIDAD EN MNIST")
    print("=" * 70)

    # 1. Cargar datos
    images, labels = load_mnist()
    images = images.to(DEVICE)

    # 2. Calcular intensidad total por imagen
    print("\nCalculando intensidad total por imagen...")
    intensities = compute_total_intensity(images)

    # 3. Calcular desviaciones del centro de gravedad (reutilizado del análisis anterior)
    # Necesitamos esto solo para el plot opcional de intensidad vs traslación
    H, W = 28, 28
    x_coords = torch.arange(W, dtype=torch.float32, device=DEVICE).view(1, W)
    y_coords = torch.arange(H, dtype=torch.float32, device=DEVICE).view(H, 1)
    img_batch = images.squeeze(1)
    mass = img_batch.sum(dim=(1, 2))
    mass = torch.clamp(mass, min=1e-8)
    cx = (img_batch * x_coords).sum(dim=(1, 2)) / mass
    cy = (img_batch * y_coords).sum(dim=(1, 2)) / mass
    centers = torch.stack([cx, cy], dim=1)
    CENTER = (28 - 1) / 2.0
    geom = torch.tensor([CENTER, CENTER], dtype=torch.float32, device=DEVICE)
    deviations = centers - geom

    # 4. Estadísticas
    print("Generando estadísticas...")
    stats = compute_statistics(intensities, labels)

    # 5. Guardar JSON
    json_path = os.path.join(RESULTS_DIR, "intensity_stats.json")
    with open(json_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nEstadísticas guardadas en: {json_path}")

    # 6. Resumen en consola
    print("\n" + "=" * 70)
    print("RESUMEN GLOBAL")
    print("=" * 70)
    g = stats['global']
    print(f"  Total de imágenes: {g['count']}")
    print(f"  Intensidad media: {g['mean']:.2f}  std: {g['std']:.2f}")
    print(f"  Rango: [{g['min']:.2f}, {g['max']:.2f}]")
    print(f"  Percentiles: P5={g['p5']:.2f}, P50={g['p50']:.2f}, P95={g['p95']:.2f}")

    print("\n" + "-" * 70)
    print("INTENSIDAD PROMEDIO POR DÍGITO")
    print("-" * 70)
    print(f"{'Dig':>4} {'N':>6} {'Media':>10} {'Std':>8} {'Min':>8} {'Max':>8}")
    for d in range(10):
        s = stats['by_digit'][str(d)]
        print(f"{d:>4} {s['count']:>6} {s['mean']:>10.2f} {s['std']:>8.2f} "
              f"{s['min']:>8.2f} {s['max']:>8.2f}")
    print("-" * 70)

    # Ordenar dígitos por intensidad media
    sorted_by_intensity = sorted(
        [(d, stats['by_digit'][str(d)]['mean']) for d in range(10)],
        key=lambda x: x[1]
    )
    print("\nDígitos ordenados por intensidad (menor a mayor):")
    for d, m in sorted_by_intensity:
        print(f"  {d}: {m:.2f}")

    # 7. Visualizaciones
    print("\nGenerando visualizaciones...")
    plot_intensity_histogram(intensities, labels, os.path.join(RESULTS_DIR, "intensity_histogram.png"))
    plot_boxplots_by_digit(intensities, labels, os.path.join(RESULTS_DIR, "intensity_boxplots.png"))
    plot_density_by_digit(intensities, labels, os.path.join(RESULTS_DIR, "intensity_density_by_digit.png"))
    plot_examples_by_intensity(images, intensities, labels, os.path.join(RESULTS_DIR, "intensity_examples.png"))
    plot_intensity_vs_translation(intensities, deviations, labels, os.path.join(RESULTS_DIR, "intensity_vs_translation.png"))

    print(f"\nTodas las figuras guardadas en: {RESULTS_DIR}")
    print("\n" + "=" * 70)
    print("ANÁLISIS COMPLETADO")
    print("=" * 70)


if __name__ == "__main__":
    main()
