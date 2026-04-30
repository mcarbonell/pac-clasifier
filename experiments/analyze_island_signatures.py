"""
Análisis de firmas de islas en MNIST.

Para cada imagen, se calcula:
  - Firma horizontal: número de islas (grupos consecutivos de píxeles activos)
    por cada fila (28 valores).
  - Firma vertical: número de islas por cada columna (28 valores).

Una "isla" es un grupo de píxeles con valor > 0 consecutivos en una fila/columna.
Por ejemplo, [0,0,1,1,0,1,0] tiene 2 islas.

Este análisis captura la estructura morfológica de cada dígito y debería
producir firmas características (ej: '0' tiene 2 islas en filas centrales,
'1' tiene típicamente 1 isla, etc.).
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
THRESHOLD = 0.0  # Píxeles > 0 se consideran activos
RESULTS_DIR = "../results/island_signatures"

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


def count_islands_1d(binary_seq):
    """
    Cuenta islas en secuencias 1D (batch).

    Parameters
    ----------
    binary_seq : torch.Tensor, shape (N, L)
        Secuencias binarias (0/1).

    Returns
    -------
    counts : torch.Tensor, shape (N,)
        Número de islas por secuencia.
    """
    # Diferencia entre elementos consecutivos
    diff = binary_seq[:, 1:] - binary_seq[:, :-1]  # (N, L-1)
    # Cada transición 0→1 es el inicio de una isla
    starts = (diff == 1).sum(dim=1)
    # Si el primer elemento es 1, es el inicio de la primera isla
    first = binary_seq[:, 0]
    return starts + first


def compute_horizontal_signatures(images, threshold=THRESHOLD):
    """
    Calcula firmas horizontales: número de islas por fila.

    Parameters
    ----------
    images : torch.Tensor, shape (N, 1, H, W)
    threshold : float
        Umbral para considerar un píxel activo.

    Returns
    -------
    signatures : torch.Tensor, shape (N, H)
        Número de islas en cada fila para cada imagen.
    """
    N, _, H, W = images.shape
    # Binarizar: (N, H, W)
    binary = (images.squeeze(1) > threshold).float()
    # Reorganizar como (N*H, W) para procesar filas como secuencias 1D
    rows = binary.view(N * H, W)
    # Contar islas por fila
    island_counts = count_islands_1d(rows)  # (N*H,)
    # Reorganizar de vuelta a (N, H)
    return island_counts.view(N, H)


def compute_vertical_signatures(images, threshold=THRESHOLD):
    """
    Calcula firmas verticales: número de islas por columna.

    Parameters
    ----------
    images : torch.Tensor, shape (N, 1, H, W)

    Returns
    -------
    signatures : torch.Tensor, shape (N, W)
        Número de islas en cada columna para cada imagen.
    """
    N, _, H, W = images.shape
    # Binarizar
    binary = (images.squeeze(1) > threshold).float()
    # Transponer para tratar columnas como filas: (N, W, H)
    cols = binary.transpose(1, 2).reshape(N * W, H)
    # Contar islas por columna
    island_counts = count_islands_1d(cols)  # (N*W,)
    # Reorganizar a (N, W)
    return island_counts.view(N, W)


def compute_statistics(horiz_signatures, vert_signatures, labels):
    """
    Calcula estadísticas de firmas por dígito.
    """
    h_np = horiz_signatures.cpu().numpy()
    v_np = vert_signatures.cpu().numpy()
    labels_np = labels.cpu().numpy()

    stats = {
        'global': {
            'count': int(len(labels_np)),
            'horiz_total_mean': float(np.mean(h_np.sum(axis=1))),
            'horiz_total_std': float(np.std(h_np.sum(axis=1))),
            'vert_total_mean': float(np.mean(v_np.sum(axis=1))),
            'vert_total_std': float(np.std(v_np.sum(axis=1))),
        },
        'by_digit': {}
    }

    for digit in range(10):
        mask = labels_np == digit
        h_d = h_np[mask]  # (N_d, 28)
        v_d = v_np[mask]  # (N_d, 28)

        stats['by_digit'][str(digit)] = {
            'count': int(mask.sum()),
            'horiz_mean_per_row': h_d.mean(axis=0).tolist(),
            'horiz_std_per_row': h_d.std(axis=0).tolist(),
            'vert_mean_per_col': v_d.mean(axis=0).tolist(),
            'vert_std_per_col': v_d.std(axis=0).tolist(),
            'horiz_total_mean': float(np.mean(h_d.sum(axis=1))),
            'horiz_total_std': float(np.std(h_d.sum(axis=1))),
            'vert_total_mean': float(np.mean(v_d.sum(axis=1))),
            'vert_total_std': float(np.std(v_d.sum(axis=1))),
        }

    return stats


# =============================================================================
# VISUALIZACIONES
# =============================================================================

def plot_signature_heatmaps(stats, save_path):
    """
    Heatmaps de firmas medias horizontales y verticales por dígito.
    """
    print(f"  Generating signature heatmaps -> {os.path.basename(save_path)}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    # Firma horizontal: 10 dígitos × 28 filas
    h_matrix = np.array([stats['by_digit'][str(d)]['horiz_mean_per_row'] for d in range(10)])
    im1 = axes[0].imshow(h_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    axes[0].set_yticks(range(10))
    axes[0].set_yticklabels(range(10))
    axes[0].set_xlabel('Fila (0=arriba, 27=abajo)', fontsize=11)
    axes[0].set_ylabel('Dígito', fontsize=11)
    axes[0].set_title('Firma Horizontal Media\n(islas por fila)', fontsize=13)
    for i in range(10):
        for j in range(28):
            val = h_matrix[i, j]
            if val > 0.3:
                text_color = 'white'
            else:
                text_color = 'black'
            axes[0].text(j, i, f'{val:.1f}', ha='center', va='center',
                        fontsize=6, color=text_color)
    plt.colorbar(im1, ax=axes[0], label='Media de islas')

    # Firma vertical: 10 dígitos × 28 columnas
    v_matrix = np.array([stats['by_digit'][str(d)]['vert_mean_per_col'] for d in range(10)])
    im2 = axes[1].imshow(v_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    axes[1].set_yticks(range(10))
    axes[1].set_yticklabels(range(10))
    axes[1].set_xlabel('Columna (0=izquierda, 27=derecha)', fontsize=11)
    axes[1].set_ylabel('Dígito', fontsize=11)
    axes[1].set_title('Firma Vertical Media\n(islas por columna)', fontsize=13)
    for i in range(10):
        for j in range(28):
            val = v_matrix[i, j]
            if val > 0.3:
                text_color = 'white'
            else:
                text_color = 'black'
            axes[1].text(j, i, f'{val:.1f}', ha='center', va='center',
                        fontsize=6, color=text_color)
    plt.colorbar(im2, ax=axes[1], label='Media de islas')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_signature_lines(stats, save_path):
    """
    Gráfico de líneas mostrando la firma media de cada dígito.
    """
    print(f"  Generating signature lines -> {os.path.basename(save_path)}")

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Horizontal
    for digit in range(10):
        h_mean = np.array(stats['by_digit'][str(digit)]['horiz_mean_per_row'])
        axes[0].plot(range(28), h_mean, label=str(digit),
                     color=plt.cm.tab10(digit / 9.0), linewidth=2, marker='o', markersize=4)
    axes[0].set_xlabel('Fila', fontsize=11)
    axes[0].set_ylabel('Media de islas', fontsize=11)
    axes[0].set_title('Firmas Horizontales Medias por Dígito', fontsize=13)
    axes[0].legend(title='Dígito', loc='upper right')
    axes[0].set_xticks(range(0, 28, 2))
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-0.1, 2.5)

    # Vertical
    for digit in range(10):
        v_mean = np.array(stats['by_digit'][str(digit)]['vert_mean_per_col'])
        axes[1].plot(range(28), v_mean, label=str(digit),
                     color=plt.cm.tab10(digit / 9.0), linewidth=2, marker='o', markersize=4)
    axes[1].set_xlabel('Columna', fontsize=11)
    axes[1].set_ylabel('Media de islas', fontsize=11)
    axes[1].set_title('Firmas Verticales Medias por Dígito', fontsize=13)
    axes[1].legend(title='Dígito', loc='upper right')
    axes[1].set_xticks(range(0, 28, 2))
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-0.1, 2.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_total_islands_distribution(horiz_signatures, vert_signatures, labels, save_path):
    """
    Distribución del número total de islas horizontales y verticales por dígito.
    """
    print(f"  Generating total islands distribution -> {os.path.basename(save_path)}")

    h_totals = horiz_signatures.sum(dim=1).cpu().numpy()
    v_totals = vert_signatures.sum(dim=1).cpu().numpy()
    labels_np = labels.cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Boxplots de totales horizontales
    data_h = [h_totals[labels_np == d] for d in range(10)]
    bp1 = axes[0].boxplot(data_h, tick_labels=range(10), patch_artist=True,
                          showfliers=False,
                          medianprops=dict(color='red', linewidth=2))
    for patch, color in zip(bp1['boxes'], plt.cm.tab10(np.arange(10) / 9.0)):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[0].set_xlabel('Dígito', fontsize=11)
    axes[0].set_ylabel('Total de islas horizontales', fontsize=11)
    axes[0].set_title('Distribución Total de Islas Horizontales', fontsize=13)

    # Boxplots de totales verticales
    data_v = [v_totals[labels_np == d] for d in range(10)]
    bp2 = axes[1].boxplot(data_v, tick_labels=range(10), patch_artist=True,
                          showfliers=False,
                          medianprops=dict(color='red', linewidth=2))
    for patch, color in zip(bp2['boxes'], plt.cm.tab10(np.arange(10) / 9.0)):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[1].set_xlabel('Dígito', fontsize=11)
    axes[1].set_ylabel('Total de islas verticales', fontsize=11)
    axes[1].set_title('Distribución Total de Islas Verticales', fontsize=13)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_example_with_signature(images, horiz_signatures, vert_signatures, labels, save_path):
    """
    Muestra ejemplos de cada dígito con sus firmas visualizadas.
    """
    print(f"  Generating examples with signatures -> {os.path.basename(save_path)}")

    labels_np = labels.cpu().numpy()
    h_np = horiz_signatures.cpu().numpy()
    v_np = vert_signatures.cpu().numpy()

    fig, axes = plt.subplots(10, 3, figsize=(12, 22))

    categories = ['Menos islas\n(P10)', 'Media\n(P50)', 'Más islas\n(P90)']

    for digit in range(10):
        mask = labels_np == digit
        idxs = np.where(mask)[0]
        totals = h_np[mask].sum(axis=1) + v_np[mask].sum(axis=1)
        sorted_idx = np.argsort(totals)

        picks = [
            idxs[sorted_idx[len(sorted_idx) // 10]],      # ~P10
            idxs[sorted_idx[len(sorted_idx) // 2]],       # P50
            idxs[sorted_idx[-1 - len(sorted_idx) // 10]]  # ~P90
        ]

        for col, (cat, img_idx) in enumerate(zip(categories, picks)):
            ax = axes[digit, col]

            # Mostrar imagen
            img = images[img_idx, 0].cpu().numpy()
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)

            # Dibujar firma horizontal (barras a la izquierda)
            h_sig = h_np[img_idx]
            for row, n_islands in enumerate(h_sig):
                if n_islands > 0:
                    # Dibujar pequeños marcadores a la izquierda
                    ax.plot([-0.5 - n_islands * 0.3], [row], 'r|', markersize=4, alpha=0.7)

            # Dibujar firma vertical (barras arriba)
            v_sig = v_np[img_idx]
            for col_idx, n_islands in enumerate(v_sig):
                if n_islands > 0:
                    ax.plot([col_idx], [-0.5 - n_islands * 0.3], 'b_', markersize=4, alpha=0.7)

            ax.set_xlim(-4, 31)
            ax.set_ylim(31, -4)
            ax.set_aspect('equal')
            ax.axis('off')

            # Anotaciones
            total_h = h_sig.sum()
            total_v = v_sig.sum()
            ax.text(14, -2.5, f'H:{total_h} V:{total_v}', ha='center', fontsize=9, color='green')

            if digit == 0:
                ax.set_title(cat, fontsize=11)

        axes[digit, 0].set_ylabel(str(digit), rotation=0, fontsize=16, labelpad=20, va='center')

    # Leyenda
    handles = [
        plt.Line2D([0], [0], marker='|', color='red', linestyle='None', markersize=10, label='Islas horiz.'),
        plt.Line2D([0], [0], marker='_', color='blue', linestyle='None', markersize=10, label='Islas vert.')
    ]
    fig.legend(handles=handles, loc='upper center', ncol=2, fontsize=12, bbox_to_anchor=(0.5, 0.995))

    fig.suptitle('Ejemplos con Firmas de Islas\n(rojo=horizontal, azul=vertical)', fontsize=16, y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def print_signature_table(stats):
    """Imprime tabla resumen de firmas por dígito."""
    print("\n" + "=" * 70)
    print("FIRMAS DE ISLAS POR DÍGITO")
    print("=" * 70)
    print(f"{'Dig':>4} {'N':>6} {'H-total':>10} {'V-total':>10} {'H-max-row':>10} {'V-max-col':>10}")
    print("-" * 70)

    for d in range(10):
        s = stats['by_digit'][str(d)]
        h_mean = np.array(s['horiz_mean_per_row'])
        v_mean = np.array(s['vert_mean_per_col'])
        print(f"{d:>4} {s['count']:>6} {s['horiz_total_mean']:>10.2f} {s['vert_total_mean']:>10.2f} "
              f"{h_mean.max():>10.2f} {v_mean.max():>10.2f}")

    print("-" * 70)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("ANÁLISIS DE FIRMAS DE ISLAS EN MNIST")
    print("=" * 70)

    # 1. Cargar datos
    images, labels = load_mnist()
    images = images.to(DEVICE)

    # 2. Calcular firmas
    print("\nCalculando firmas horizontales (islas por fila)...")
    horiz_sigs = compute_horizontal_signatures(images)

    print("Calculando firmas verticales (islas por columna)...")
    vert_sigs = compute_vertical_signatures(images)

    # 3. Estadísticas
    print("Generando estadísticas...")
    stats = compute_statistics(horiz_sigs, vert_sigs, labels)

    # 4. Guardar JSON
    json_path = os.path.join(RESULTS_DIR, "island_signatures_stats.json")
    with open(json_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nEstadísticas guardadas en: {json_path}")

    # 5. Tabla resumen
    print_signature_table(stats)

    # 6. Visualizaciones
    print("\nGenerando visualizaciones...")
    plot_signature_heatmaps(stats, os.path.join(RESULTS_DIR, "signature_heatmaps.png"))
    plot_signature_lines(stats, os.path.join(RESULTS_DIR, "signature_lines.png"))
    plot_total_islands_distribution(horiz_sigs, vert_sigs, labels,
                                    os.path.join(RESULTS_DIR, "total_islands_distribution.png"))
    plot_example_with_signature(images, horiz_sigs, vert_sigs, labels,
                                os.path.join(RESULTS_DIR, "examples_with_signatures.png"))

    print(f"\nTodas las figuras guardadas en: {RESULTS_DIR}")
    print("\n" + "=" * 70)
    print("ANÁLISIS COMPLETADO")
    print("=" * 70)


if __name__ == "__main__":
    main()
