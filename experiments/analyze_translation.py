"""
Análisis de traslación en MNIST mediante centro de gravedad.

Este script calcula, para cada imagen del dataset MNIST, el centro de gravedad
ponderado por la intensidad de los píxeles. Luego mide cuánto se desvía ese
centro del centro geométrico de la imagen (13.5, 13.5 para 28×28).

El objetivo es cuantificar la traslación sistemática de los dígitos dentro
del canvas, como paso previo a experimentar con normalización por centro de
masa para mejorar el clasificador PAC.
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
IMG_SIZE = 28
CENTER = (IMG_SIZE - 1) / 2.0  # 13.5 para 28×28
RESULTS_DIR = "../results/translation_analysis"

# Crear directorio de resultados
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
    x_train, y_train = next(iter(train_loader))  # (60000, 1, 28, 28)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10000, shuffle=False)
    x_test, y_test = next(iter(test_loader))      # (10000, 1, 28, 28)

    # Concatenar train + test para análisis global
    x_all = torch.cat([x_train, x_test], dim=0)   # (70000, 1, 28, 28)
    y_all = torch.cat([y_train, y_test], dim=0)   # (70000,)

    print(f"Total images loaded: {x_all.shape[0]}")
    return x_all, y_all


def compute_center_of_gravity(images):
    """
    Calcula el centro de gravedad (Cx, Cy) para un lote de imágenes.

    Parameters
    ----------
    images : torch.Tensor, shape (N, 1, H, W)
        Lote de imágenes en escala de grises.

    Returns
    -------
    centers : torch.Tensor, shape (N, 2)
        Coordenadas (Cx, Cy) del centro de gravedad de cada imagen.
    """
    N, _, H, W = images.shape
    device = images.device

    # Crear grillas de coordenadas: x en [0, W-1], y en [0, H-1]
    x_coords = torch.arange(W, dtype=torch.float32, device=device).view(1, W)  # (1, W)
    y_coords = torch.arange(H, dtype=torch.float32, device=device).view(H, 1)  # (H, 1)

    # Squeeze canal para trabajar con (N, H, W)
    img = images.squeeze(1)  # (N, H, W)

    # Masa total por imagen: suma de intensidades
    mass = img.sum(dim=(1, 2))  # (N,)

    # Evitar división por cero (imágenes completamente negras)
    mass = torch.clamp(mass, min=1e-8)

    # Centro de gravedad en X: sum(y, x * I) / sum(I)
    # Usamos broadcasting: img (N,H,W) * x_coords (1,W) -> (N,H,W)
    cx = (img * x_coords).sum(dim=(1, 2)) / mass  # (N,)

    # Centro de gravedad en Y: sum(y, y * I) / sum(I)
    cy = (img * y_coords).sum(dim=(1, 2)) / mass  # (N,)

    return torch.stack([cx, cy], dim=1)  # (N, 2)


def compute_deviations(centers, center_geom=CENTER):
    """
    Calcula la desviación del centro de gravedad respecto al centro geométrico.

    Returns
    -------
    deviations : torch.Tensor, shape (N, 2)
        (dx, dy) en píxeles.
    """
    geom = torch.tensor([center_geom, center_geom], dtype=torch.float32, device=centers.device)
    return centers - geom  # (N, 2)


def compute_statistics(deviations, labels):
    """
    Calcula estadísticas globales y por dígito.

    Returns
    -------
    stats : dict
        Diccionario con estadísticas.
    """
    dx = deviations[:, 0].cpu().numpy()
    dy = deviations[:, 1].cpu().numpy()
    labels_np = labels.cpu().numpy()

    stats = {
        'global': {
            'count': int(len(dx)),
            'dx_mean': float(np.mean(dx)),
            'dx_std': float(np.std(dx)),
            'dx_min': float(np.min(dx)),
            'dx_max': float(np.max(dx)),
            'dx_p5': float(np.percentile(dx, 5)),
            'dx_p25': float(np.percentile(dx, 25)),
            'dx_p50': float(np.percentile(dx, 50)),
            'dx_p75': float(np.percentile(dx, 75)),
            'dx_p95': float(np.percentile(dx, 95)),
            'dy_mean': float(np.mean(dy)),
            'dy_std': float(np.std(dy)),
            'dy_min': float(np.min(dy)),
            'dy_max': float(np.max(dy)),
            'dy_p5': float(np.percentile(dy, 5)),
            'dy_p25': float(np.percentile(dy, 25)),
            'dy_p50': float(np.percentile(dy, 50)),
            'dy_p75': float(np.percentile(dy, 75)),
            'dy_p95': float(np.percentile(dy, 95)),
        },
        'by_digit': {}
    }

    for digit in range(10):
        mask = labels_np == digit
        dx_d = dx[mask]
        dy_d = dy[mask]

        stats['by_digit'][str(digit)] = {
            'count': int(mask.sum()),
            'dx_mean': float(np.mean(dx_d)),
            'dx_std': float(np.std(dx_d)),
            'dx_min': float(np.min(dx_d)),
            'dx_max': float(np.max(dx_d)),
            'dx_p5': float(np.percentile(dx_d, 5)),
            'dx_p50': float(np.percentile(dx_d, 50)),
            'dx_p95': float(np.percentile(dx_d, 95)),
            'dy_mean': float(np.mean(dy_d)),
            'dy_std': float(np.std(dy_d)),
            'dy_min': float(np.min(dy_d)),
            'dy_max': float(np.max(dy_d)),
            'dy_p5': float(np.percentile(dy_d, 5)),
            'dy_p50': float(np.percentile(dy_d, 50)),
            'dy_p95': float(np.percentile(dy_d, 95)),
        }

    return stats


# =============================================================================
# VISUALIZACIONES
# =============================================================================

def plot_heatmap_2d(deviations, labels, save_path):
    """Heatmap 2D de dx vs dy, con histogramas marginales por clase."""
    print(f"  Generating 2D heatmap -> {os.path.basename(save_path)}")
    dx = deviations[:, 0].cpu().numpy()
    dy = deviations[:, 1].cpu().numpy()
    labels_np = labels.cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 12),
                             gridspec_kw={'width_ratios': [4, 1], 'height_ratios': [1, 4]})

    # Colores por dígito
    colors = plt.cm.tab10(labels_np / 9.0)

    # Scatter principal (abajo-izquierda)
    ax_main = axes[1, 0]
    for digit in range(10):
        mask = labels_np == digit
        ax_main.scatter(dx[mask], dy[mask], s=1, alpha=0.3, label=str(digit), color=plt.cm.tab10(digit / 9.0))
    ax_main.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax_main.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax_main.set_xlabel('Desviación X (píxeles)', fontsize=12)
    ax_main.set_ylabel('Desviación Y (píxeles)', fontsize=12)
    ax_main.set_title('Centro de Gravedad: Desviación respecto al Centro Geométrico', fontsize=14)
    ax_main.legend(title='Dígito', loc='upper right', markerscale=5)
    ax_main.set_xlim(-14, 14)
    ax_main.set_ylim(-14, 14)
    ax_main.set_aspect('equal')

    # Histograma marginal superior (dx)
    ax_top = axes[0, 0]
    ax_top.hist(dx, bins=100, color='steelblue', edgecolor='white', alpha=0.7)
    ax_top.axvline(0, color='red', linestyle='--', alpha=0.7)
    ax_top.set_xlim(-14, 14)
    ax_top.set_xticks([])
    ax_top.set_ylabel('Freq')

    # Histograma marginal derecho (dy)
    ax_right = axes[1, 1]
    ax_right.hist(dy, bins=100, orientation='horizontal', color='steelblue', edgecolor='white', alpha=0.7)
    ax_right.axhline(0, color='red', linestyle='--', alpha=0.7)
    ax_right.set_ylim(-14, 14)
    ax_right.set_yticks([])
    ax_right.set_xlabel('Freq')

    # Esconder esquina vacía
    axes[0, 1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_boxplots_by_digit(deviations, labels, save_path):
    """Boxplots de dx y dy agrupados por dígito."""
    print(f"  Generating boxplots -> {os.path.basename(save_path)}")
    dx = deviations[:, 0].cpu().numpy()
    dy = deviations[:, 1].cpu().numpy()
    labels_np = labels.cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    data_dx = [dx[labels_np == d] for d in range(10)]
    data_dy = [dy[labels_np == d] for d in range(10)]

    bp1 = axes[0].boxplot(data_dx, labels=range(10), patch_artist=True,
                          showfliers=False,  # Ocultar outliers para claridad
                          medianprops=dict(color='red', linewidth=2))
    for patch, color in zip(bp1['boxes'], plt.cm.tab10(np.arange(10) / 9.0)):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[0].axhline(0, color='black', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Dígito')
    axes[0].set_ylabel('Desviación X (píxeles)')
    axes[0].set_title('Desviación en Eje X por Dígito')

    bp2 = axes[1].boxplot(data_dy, labels=range(10), patch_artist=True,
                          showfliers=False,
                          medianprops=dict(color='red', linewidth=2))
    for patch, color in zip(bp2['boxes'], plt.cm.tab10(np.arange(10) / 9.0)):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[1].axhline(0, color='black', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Dígito')
    axes[1].set_ylabel('Desviación Y (píxeles)')
    axes[1].set_title('Desviación en Eje Y por Dígito')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_example_images_with_centers(images, centers, labels, save_path, n_examples=10):
    """
    Muestra una cuadrícula con ejemplos de cada dígito, marcando el centro
    geométrico (rojo) y el centro de gravedad (verde).
    """
    print(f"  Generating example grid -> {os.path.basename(save_path)}")
    labels_np = labels.cpu().numpy()
    centers_np = centers.cpu().numpy()

    fig, axes = plt.subplots(10, n_examples, figsize=(n_examples * 1.5, 15))

    for digit in range(10):
        mask = labels_np == digit
        idxs = np.where(mask)[0]
        # Elegir ejemplos variados: extremos en dx
        dxs = centers_np[idxs, 0] - CENTER
        sorted_idx = np.argsort(dxs)
        # Tomar 2 de la izquierda, algunos del centro, 2 de la derecha
        picks = [
            idxs[sorted_idx[0]], idxs[sorted_idx[1]],                    # más a la izq
            idxs[sorted_idx[len(sorted_idx)//3]],
            idxs[sorted_idx[len(sorted_idx)//2]],
            idxs[sorted_idx[2*len(sorted_idx)//3]],
            idxs[sorted_idx[-2]], idxs[sorted_idx[-1]]                   # más a la der
        ]
        # Si n_examples > len(picks), rellenar con aleatorios
        while len(picks) < n_examples:
            picks.append(np.random.choice(idxs))
        picks = picks[:n_examples]

        for col, img_idx in enumerate(picks):
            ax = axes[digit, col] if n_examples > 1 else axes[digit]
            img = images[img_idx, 0].cpu().numpy()
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            ax.axis('off')

            # Centro geométrico (rojo)
            ax.plot(CENTER, CENTER, 'r+', markersize=12, markeredgewidth=2, label='Centro geométrico')

            # Centro de gravedad (verde)
            cx, cy = centers_np[img_idx]
            ax.plot(cx, cy, 'go', markersize=8, markeredgecolor='lime', markeredgewidth=2, label='Centro de gravedad')

            # Línea de desviación
            ax.plot([CENTER, cx], [CENTER, cy], 'y-', linewidth=1.5, alpha=0.7)

            # Título solo en la primera fila
            if digit == 0:
                ax.set_title(f'Ej. {col+1}', fontsize=10)

        # Etiqueta de fila
        axes[digit, 0].set_ylabel(str(digit), rotation=0, fontsize=16, labelpad=20, va='center')

    # Leyenda única
    handles = [
        plt.Line2D([0], [0], marker='+', color='red', linestyle='None', markersize=10, label='Centro geométrico'),
        plt.Line2D([0], [0], marker='o', color='green', linestyle='None', markersize=8, label='Centro de gravedad'),
        plt.Line2D([0], [0], color='yellow', linewidth=2, label='Desviación')
    ]
    fig.legend(handles=handles, loc='upper center', ncol=3, fontsize=12, bbox_to_anchor=(0.5, 0.98))

    fig.suptitle('Ejemplos de Desplazamiento del Centro de Gravedad\n(× = centro geométrico, ● = centro de gravedad)', fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_density_comparison(deviations, labels, save_path):
    """Compara la densidad de desviaciones dx y dy globalmente."""
    print(f"  Generating density plot -> {os.path.basename(save_path)}")
    dx = deviations[:, 0].cpu().numpy()
    dy = deviations[:, 1].cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(dx, bins=80, density=True, alpha=0.4, color='steelblue', label='Desviación X', edgecolor='white')
    ax.hist(dy, bins=80, density=True, alpha=0.4, color='coral', label='Desviación Y', edgecolor='white')
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Desviación (píxeles)')
    ax.set_ylabel('Densidad')
    ax.set_title('Distribución de Desviaciones del Centro de Gravedad')
    ax.legend()
    ax.set_xlim(-8, 8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()



# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("ANÁLISIS DE TRASLACIÓN EN MNIST (CENTRO DE GRAVEDAD)")
    print("=" * 70)

    # 1. Cargar datos
    images, labels = load_mnist()
    images = images.to(DEVICE)

    # 2. Calcular centros de gravedad
    print("\nCalculando centros de gravedad...")
    centers = compute_center_of_gravity(images)  # (N, 2)

    # 3. Calcular desviaciones
    print("Calculando desviaciones respecto al centro geométrico...")
    deviations = compute_deviations(centers)  # (N, 2)

    # 4. Estadísticas
    print("Generando estadísticas...")
    stats = compute_statistics(deviations, labels)

    # 5. Guardar estadísticas en JSON
    json_path = os.path.join(RESULTS_DIR, "translation_stats.json")
    with open(json_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nEstadísticas guardadas en: {json_path}")

    # 6. Imprimir resumen en consola
    print("\n" + "=" * 70)
    print("RESUMEN GLOBAL")
    print("=" * 70)
    g = stats['global']
    print(f"  Total de imágenes: {g['count']}")
    print(f"  Desviación X ->  media: {g['dx_mean']:+.3f}  std: {g['dx_std']:.3f}  "
          f"min: {g['dx_min']:+.3f}  max: {g['dx_max']:+.3f}")
    print(f"  Desviación Y ->  media: {g['dy_mean']:+.3f}  std: {g['dy_std']:.3f}  "
          f"min: {g['dy_min']:+.3f}  max: {g['dy_max']:+.3f}")

    print("\n" + "-" * 70)
    print("ESTADÍSTICAS POR DÍGITO")
    print("-" * 70)
    print(f"{'Dig':>4} {'N':>6} {'dx_mean':>10} {'dx_std':>8} {'dy_mean':>10} {'dy_std':>8}")
    for d in range(10):
        s = stats['by_digit'][str(d)]
        print(f"{d:>4} {s['count']:>6} {s['dx_mean']:>+10.3f} {s['dx_std']:>8.3f} "
              f"{s['dy_mean']:>+10.3f} {s['dy_std']:>8.3f}")
    print("-" * 70)

    # 7. Generar visualizaciones
    print("\nGenerando visualizaciones...")
    plot_heatmap_2d(deviations, labels, os.path.join(RESULTS_DIR, "heatmap_2d_deviations.png"))
    plot_boxplots_by_digit(deviations, labels, os.path.join(RESULTS_DIR, "boxplots_by_digit.png"))
    plot_example_images_with_centers(images, centers, labels, os.path.join(RESULTS_DIR, "examples_with_centers.png"))
    plot_density_comparison(deviations, labels, os.path.join(RESULTS_DIR, "density_comparison.png"))

    print(f"\nTodas las figuras guardadas en: {RESULTS_DIR}")
    print("\n" + "=" * 70)
    print("ANÁLISIS COMPLETADO")
    print("=" * 70)


if __name__ == "__main__":
    main()
