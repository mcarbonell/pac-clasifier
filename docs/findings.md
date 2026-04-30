# Hallazgos de Experimentos MNIST — PAC Classifier

Este documento recopila los resultados y análisis de cuatro experimentos realizados sobre el dataset MNIST con el objetivo de caracterizar propiedades geométricas, morfológicas y estructurales de los dígitos escritos a mano, y explorar representaciones alternativas para la clasificación con el algoritmo PAC (Purifying Archetype Classifier).

---

## Tabla Resumen

| Experimento | Script | Resultado Clave |
|-------------|--------|-----------------|
| Análisis de Traslación | `analyze_translation.py` | Sesgo sistemático de ~+0.5 píxeles en X e Y |
| Análisis de Intensidad | `analyze_intensity.py` | '0' es el dígito más pesado (135.8), '1' el más ligero (59.7) |
| Firmas de Islas | `analyze_island_signatures.py` | Cada dígito posee una huella morfológica única e identificable |
| Clasificador PAC + Firmas | `pac_island_signature_classifier.py` | 85.30% accuracy con 56D (vs 96.07% con 784D) |

---

## Experimento 1: Análisis de Traslación (Centro de Gravedad)

### 1.1 Objetivo

Determinar si existe un **sesgo sistemático en la posición** de los dígitos dentro del canvas de 28×28 píxeles. La hipótesis es que los dígitos no están perfectamente centrados, sino que presentan una traslación preferente en alguna dirección, lo cual podría afectar la clasificación basada en distancias como la del algoritmo PAC.

### 1.2 Metodología

Para cada imagen del dataset MNIST (70,000 imágenes: 60,000 train + 10,000 test):

1. **Centro de gravedad ponderado**: calcular el centro de masa donde cada píxel contribuye proporcionalmente a su intensidad.
   $$C_x = \\frac{\\sum_{x,y} x \\cdot I(x,y)}{\\sum_{x,y} I(x,y)}, \\quad C_y = \\frac{\\sum_{x,y} y \\cdot I(x,y)}{\\sum_{x,y} I(x,y)}$$

2. **Desviación respecto al centro geométrico**: el centro teórico de una imagen 28×28 es (13.5, 13.5).
   $$\\Delta x = C_x - 13.5, \\quad \\Delta y = C_y - 13.5$$

3. **Estadísticas globales y por dígito**: media, desviación estándar, percentiles 5/25/50/75/95, mínimo y máximo.

4. **Visualizaciones**: heatmap 2D de desviaciones, boxplots por dígito, ejemplos con centros marcados.

### 1.3 Implementación

- **Script**: `experiments/analyze_translation.py`
- **Salidas**:
  - `results/translation_analysis/translation_stats.json`
  - `results/translation_analysis/heatmap_2d_deviations.png`
  - `results/translation_analysis/boxplots_by_digit.png`
  - `results/translation_analysis/examples_with_centers.png`
  - `results/translation_analysis/density_comparison.png`

### 1.4 Resultados

**Estadísticas globales (70,000 imágenes):**

| Métrica | Valor |
|---------|-------|
| `dx_mean` | **+0.507** |
| `dx_std` | 0.356 |
| `dx_min` | -0.95 |
| `dx_max` | +1.00 |
| `dy_mean` | **+0.496** |
| `dy_std` | 0.339 |
| `dy_min` | -0.78 |
| `dy_max` | +1.00 |

**Por dígito (media de desviaciones):**

| Dígito | dx_mean | dy_mean | Observación |
|--------|---------|---------|-------------|
| 0 | +0.40 | +0.38 | Cercano al centro |
| 1 | +0.28 | +0.30 | Muy centrado |
| 2 | +0.66 | +0.54 | Desplazado ↘ |
| 3 | +0.66 | +0.56 | Desplazado ↘ |
| 4 | +0.34 | +0.61 | Desplazado ↓ |
| 5 | +0.63 | +0.55 | Desplazado ↘ |
| 6 | +0.49 | +0.42 | Cercano al centro |
| 7 | +0.32 | +0.58 | Desplazado ↓ |
| 8 | +0.48 | +0.49 | Cercano al centro |
| 9 | +0.52 | +0.48 | Cercano al centro |

### 1.5 Análisis

1. **Sesgo sistemático hacia abajo-derecha**: tanto la media global de `dx` (+0.507) como la de `dy` (+0.496) son positivas y significativas. Esto indica que, en promedio, los dígitos en MNIST están **ligeramente desplazados hacia la esquina inferior derecha** del canvas.

2. **El dígito '1' es el más centrado**: con desviaciones de (+0.28, +0.30), el '1' tiende a estar más centrado que otros dígitos. Esto tiene sentido porque los escritores de '1' suelen trazar un trazo vertical simple sin extenderse mucho horizontalmente.

3. **Los dígitos '2', '3', '5' son los más desplazados**: presentan las mayores desviaciones en ambos ejes. Esto se debe a que estos dígitos tienen trazos horizontales en la parte superior (como el '7' inicial del '2') que extienden el centro de gravedad hacia la derecha, y curvas que lo bajan.

4. **La desviación máxima es de ~1 píxel**: aunque el sesgo es sistemático, es relativamente pequeño (menos de 1 píxel en promedio). Sin embargo, en tareas de clasificación de alta precisión, incluso desplazamientos sub-píxel pueden afectar la similitud del coseno entre vectores de 784 dimensiones.

### 1.6 Conclusiones

- Existe un **sesgo de traslación en MNIST** que podría estar introduciendo ruido en clasificadores basados en distancia.
- Una **normalización por centro de gravedad** (desplazar cada imagen para que su centro de masa coincida con el centro geométrico) podría mejorar la alineación del dataset.
- Este experimento sienta las bases para un futuro experimento de normalización previa al entrenamiento del clasificador PAC.

---

## Experimento 2: Análisis de Intensidad de Píxeles

### 2.1 Objetivo

Cuantificar la **\"densidad de tinta\"** de cada dígito — es decir, la suma total de intensidades de píxeles — para entender qué tan \"pesados\" o \"ligeros\" son los diferentes dígitos en términos de área ocupada y grosor de trazo.

### 2.2 Metodología

1. **Intensidad total**: para cada imagen, sumar todos los valores de píxeles (escala 0.0–1.0 después de `ToTensor()`, multiplicados por 255 para obtener valores en escala 0–255).

2. **Estadísticas por dígito**: media, std, percentiles, min/max.

3. **Visualizaciones**: histogramas globales, boxplots por dígito, densidades, ejemplos extremos.

4. **Correlación con traslación**: analizar si los dígitos más intensos tienden a estar más o menos desplazados.

### 2.3 Implementación

- **Script**: `experiments/analyze_intensity.py`
- **Salidas**:
  - `results/intensity_analysis/intensity_stats.json`
  - `results/intensity_analysis/histogram_global.png`
  - `results/intensity_analysis/boxplots_by_digit.png`
  - `results/intensity_analysis/density_by_digit.png`
  - `results/intensity_analysis/examples_by_intensity.png`
  - `results/intensity_analysis/intensity_vs_translation.png`

### 2.4 Resultados

**Estadísticas globales:**

| Métrica | Valor (0–255) |
|---------|---------------|
| Media | **102.65** |
| Std | 43.46 |
| Mínimo | 30.10 |
| Máximo | 255.00 |

**Intensidad media por dígito:**

| Dígito | Intensidad Media | Ranking |
|--------|-----------------|---------|
| **0** | **135.82** | Más pesado |
| 8 | 127.27 |
| 6 | 115.87 |
| 4 | 113.60 |
| 2 | 107.82 |
| 9 | 106.90 |
| 3 | 104.23 |
| 5 | 97.21 |
| 7 | 89.22 |
| **1** | **59.67** | Más ligero |

### 2.5 Análisis

1. **El '0' es el dígito más \"pesado\"**: con una intensidad media de 135.8, el '0' ocupa significativamente más área que otros dígitos. Esto se debe a que es un círculo cerrado que ocupa gran parte del canvas.

2. **El '1' es el más \"ligero\"**: con solo 59.67 de intensidad media, el '1' ocupa menos de la mitad del área que el '0'. Esto es consistente con ser un simple trazo vertical.

3. **Dígitos con cierres (0, 6, 8, 9) son más pesados**: los dígitos que contienen bucles cerrados tienden a acumular más intensidad debido al mayor área encerrada.

4. **Dígitos con trazos verticales (1, 7) son más ligeros**: ocupan menos área horizontal y, por tanto, tienen menor intensidad total.

5. **Alta variabilidad intra-clase**: la desviación estándar es grande (~40–50), indicando que dentro de cada dígito hay mucha variación en grosor de trazo y tamaño.

### 2.6 Conclusiones

- La intensidad total es una característica **parcialmente discriminativa**: puede ayudar a distinguir '0' de '1', pero no es suficiente por sí sola debido a la gran superposición entre clases.
- Podría usarse como **característica adicional** en un clasificador multinomial o como filtro previo.
- La correlación con la traslación es débil: los dígitos más pesados no están sistemáticamente más desplazados.

---

## Experimento 3: Firmas de Islas (Island Signatures)

### 3.1 Objetivo

Extraer una **huella morfológica** de cada dígito basada en el número de grupos conectados de píxeles activos (\"islas\") en cada fila (horizontal) y cada columna (vertical). La hipótesis es que cada dígito tiene un patrón característico de islas que lo distingue de los demás.

### 3.2 Metodología

Para cada imagen binarizada (píxel > 0):

1. **Islas horizontales**: para cada fila, contar cuántos grupos consecutivos de píxeles activos existen.
   - Ejemplo: `..##..#...` → 2 islas (`##` y `#`)

2. **Islas verticales**: para cada columna, contar cuántos grupos consecutivos de píxeles activos existen.

3. **Vector de firma**: concatenar los 28 valores horizontales + 28 valores verticales = **vector de 56 dimensiones**.

4. **Estadísticas**: media y std por fila/columna para cada dígito.

5. **Visualizaciones**: heatmaps de firmas medias, gráficos de líneas, distribuciones totales.

### 3.3 Implementación

- **Script**: `experiments/analyze_island_signatures.py`
- **Salidas**:
  - `results/island_signatures/island_signatures_stats.json`
  - `results/island_signatures/signature_heatmaps.png`
  - `results/island_signatures/signature_lines.png`
  - `results/island_signatures/total_islands_distribution.png`
  - `results/island_signatures/examples_with_signatures.png`

### 3.4 Resultados

**Totales globales:**

| Dirección | Media | Std |
|-----------|-------|-----|
| Horizontal | 25.24 | 4.35 |
| Vertical | 25.09 | 8.58 |

**Totales por dígito:**

| Dígito | H-total | V-total | Firma distintiva |
|--------|---------|---------|------------------|
| **0** | **31.48** | 28.61 | Mayor firma horizontal. ~2 islas consistentes en filas centrales (lazo cerrado) |
| **1** | **20.20** | **9.80** | **Más simple**: ~1 isla por fila, muy pocas islas verticales |
| 2 | 24.38 | 30.52 | Curva compleja, alta complejidad vertical |
| **3** | 23.79 | **32.67** | **Mayor firma vertical** (dos bucles horizontales) |
| 4 | 28.14 | 21.10 | Estructura angular, segunda mayor firma horizontal |
| 5 | 22.51 | 32.50 | Similar a 3 en complejidad vertical |
| 6 | 26.34 | 24.71 | Un bucle + trazo vertical |
| 7 | 22.68 | 22.79 | Línea horizontal + diagonal |
| 8 | 27.85 | 27.78 | Dos bucles apilados, firma equilibrada |
| 9 | 25.74 | 23.01 | Similar a 6 pero invertido |

### 3.5 Análisis

1. **El '1' tiene la firma más simple**: consistentemente ~1 isla por fila y muy pocas islas verticales (~9.8 total). Esto captura perfectamente su naturaleza de trazo único vertical.

2. **El '0' tiene la firma horizontal más compleja**: alcanza 31.48 islas horizontales totales porque en las filas centrales tiene consistentemente **2 islas** (los dos lados del círculo), creando un patrón muy característico.

3. **Los dígitos '3' y '5' dominan en islas verticales** (~32.7): debido a sus múltiples segmentos horizontales que cruzan muchas columnas, generando múltiples islas al analizar verticalmente.

4. **Mayor variabilidad vertical** (std 8.58 vs 4.35 horizontal): los dígitos difieren más en su estructura horizontal (anchura, segmentos) que en su estructura vertical.

5. **Cada dígito tiene una \"huella dactilar\" morfológica**: los heatmaps de firmas medias muestran patrones claramente distinguibles. Por ejemplo, el '0' muestra un valle de 2 islas en filas 10–16, mientras que el '1' mantiene ~1 isla estable en casi todas las filas.

### 3.6 Conclusiones

- Las firmas de islas proporcionan una representación **muy compacta y altamente interpretable** de la morfología de cada dígito.
- La representación de 56 enteros captura información estructural esencial que es **invariante al grosor del trazo** (a diferencia de los píxeles brutos).
- Estas firmas podrían usarse como **características de pre-procesamiento** o para **visualización e interpretabilidad** del comportamiento del clasificador.

---

## Experimento 4: Clasificador PAC con Firmas de Islas

### 4.1 Objetivo

Evaluar si las **firmas de islas (56D)** pueden usarse como representación directa para clasificación con PAC-V2, comparando su rendimiento contra la representación tradicional de **píxeles brutos (784D)**.

### 4.2 Metodología

1. **Preparar representaciones**:
   - Firmas de islas: vector de 56 enteros (28 horizontales + 28 verticales) por imagen.
   - Píxeles brutos: vector de 784 flotantes (28×28 aplanado) por imagen.

2. **Entrenar PAC-V2** en ambas representaciones con los mismos hiperparámetros:
   - `max_iters=100`
   - `target_acc=0.999`

3. **Evaluar** en el conjunto de test (10,000 imágenes).

4. **Comparar**: accuracy, número de archetypes descubiertos, tiempo de entrenamiento.

### 4.3 Implementación

- **Script**: `experiments/pac_island_signature_classifier.py`
- **Salidas**:
  - `results/island_classifier/classifier_comparison.json`
  - `results/island_classifier/confusion_matrices.png`
  - `results/island_classifier/comparison_metrics.png`
  - `results/island_classifier/signature_archetypes.png`

### 4.4 Resultados

| Métrica | Firmas (56D) | Píxeles (784D) |
|---------|-------------|----------------|
| **Accuracy Test** | **85.30%** | **96.07%** |
| **Accuracy Train** | 89.11% | 97.70% |
| **Archetypes** | 1611 | 1470 |
| **Train Time** | 67.69s | 113.32s |
| **Dimensión** | 56 | 784 |
| **Speedup** | 1.7x | — |
| **Reducción dim.** | 14.0x | — |

### 4.5 Análisis

1. **Las firmas capturan información discriminativa**: Con solo 56 enteros por imagen, PAC-V2 logra clasificar correctamente el **85.30%** de los dígitos. Esto demuestra que la estructura de islas contiene información esencial sobre la forma de cada dígito.

2. **Pérdida de información fina**: La caída de **~10.8 puntos porcentuales** (96.07% → 85.30%) indica que las firmas de islas pierden detalles críticos necesarios para distinguir dígitos similares. Por ejemplo:
   - **4 vs 9**: ambos tienen trazos verticales y curvas; sin la posición exacta de cada píxel, la firma de islas puede confundirlos.
   - **3 vs 8**: ambos tienen múltiples bucles; la diferencia en la conectividad exacta se pierde en la firma.
   - **5 vs 3**: comparten similitudes estructurales que la firma no diferencia suficientemente.

3. **Más archetypes con menos dimensionalidad**: Curiosamente, el clasificador basado en firmas necesita **más archetypes** (1611 vs 1470). Esto sugiere que:
   - El espacio de firmas de 56D es más **ambiguo** — diferentes clases se solapan más.
   - PAC-V2 necesita más prototipos para compensar la menor cantidad de información por vector.
   - En el espacio de píxeles de 784D, cada archetype puede \"cubrir\" más ejemplos debido a la mayor riqueza de información.

4. **Trade-off claro**: Se obtiene una **reducción de 14× en dimensionalidad** y **1.7× en tiempo de entrenamiento**, a costa de ~11pp de accuracy.

5. **El entrenamiento con firmas converge más rápido** pero a un óptimo inferior:
   - Firmas: se estanca en ~89% training accuracy (generación 30+)
   - Píxeles: se estanca en ~97.7% training accuracy (generación 80+)
   - Ninguno alcanza el target de 99.9% en 100 iteraciones, lo cual es consistente con el comportamiento observado en experimentos previos del proyecto.

### 4.6 Conclusiones

- Las firmas de islas son **insuficientes por sí solas** para clasificación de alta precisión, pero demuestran que la morfología de trazo contiene información valiosa.
- **Aplicaciones potenciales**:
  - **Pre-filtrado rápido**: usar firmas para reducir el espacio de búsqueda antes de aplicar un clasificador más pesado.
  - **Interpretabilidad**: los archetypes descubiertos en el espacio de firmas son directamente interpretables como \"patrones de islas\".
  - **Fusión de características**: combinar firmas de islas con píxeles (56 + 784 = 840D) podría mejorar la robustez.
- **Normalización futura**: antes de computar firmas, aplicar la normalización por centro de gravedad (Experimento 1) podría reducir la varianza y mejorar la consistencia de las firmas.

---

## Discusión General y Líneas Futuras

### Hallazgos Transversales

1. **MNIST tiene sesgos sistemáticos**: tanto en traslación (~+0.5px) como en intensidad (grandes variaciones intra-clase). Estos sesgos son \"características no informativas\" que el clasificador podría estar aprendiendo en lugar de la verdadera forma del dígito.

2. **Representaciones compactas vs completas**: existe un **trade-off fundamental** entre la dimensionalidad de la representación y la precisión de clasificación:
   - 56D (firmas): rápida, interpretable, pero imprecisa (85%)
   - 784D (píxeles): lenta, opaca, pero precisa (96%)

3. **La morfología importa**: los experimentos 3 y 4 demuestran que la estructura de \"qué píxeles están conectados y en qué dirección\" es más importante que la intensidad exacta de cada píxel.

### Experimentos Propuestos para el Futuro

| Prioridad | Experimento | Objetivo Esperado |
|-----------|-------------|-------------------|
| Alta | **Normalización por centro de gravedad** | Eliminar el sesgo de traslación; mejorar alineación |
| Alta | **Combinar píxeles + firmas** | Explorar si 840D (784+56) mejora sobre 784D |
| Media | **Firmas con distancias normalizadas** | Incluir distancia entre islas, no solo conteo |
| Media | **Firmas en imágenes normalizadas** | Aplicar normalización de traslación antes de extraer firmas |
| Baja | **Clustering de firmas** | Usar firmas para detectar variantes estilísticas dentro de cada dígito |

---

## Apéndice: Estructura de Archivos

```
experiments/
├── analyze_translation.py              # Experimento 1: Centro de gravedad
├── analyze_intensity.py                # Experimento 2: Intensidad
├── analyze_island_signatures.py        # Experimento 3: Firmas de islas
└── pac_island_signature_classifier.py  # Experimento 4: Clasificador PAC

results/
├── translation_analysis/
│   ├── translation_stats.json
│   ├── heatmap_2d_deviations.png
│   ├── boxplots_by_digit.png
│   ├── examples_with_centers.png
│   └── density_comparison.png
├── intensity_analysis/
│   ├── intensity_stats.json
│   ├── histogram_global.png
│   ├── boxplots_by_digit.png
│   ├── density_by_digit.png
│   ├── examples_by_intensity.png
│   └── intensity_vs_translation.png
├── island_signatures/
│   ├── island_signatures_stats.json
│   ├── signature_heatmaps.png
│   ├── signature_lines.png
│   ├── total_islands_distribution.png
│   └── examples_with_signatures.png
└── island_classifier/
    ├── classifier_comparison.json
    ├── confusion_matrices.png
    ├── comparison_metrics.png
    └── signature_archetypes.png
```

---

*Documento generado automáticamente a partir de los resultados de los experimentos ejecutados sobre el dataset MNIST con el clasificador PAC.*
