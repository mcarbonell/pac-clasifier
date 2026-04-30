# TODO: Análisis de Traslación MNIST (Centro de Gravedad)

- [x] Crear directorios `experiments/` y `results/translation_analysis/`
- [x] Implementar `experiments/analyze_translation.py`:
  - Cargar MNIST (train + test)
  - Calcular centro de gravedad ponderado por intensidad
  - Calcular desviación respecto al centro geométrico (13.5, 13.5)
  - Generar estadísticas globales y por dígito
  - Crear visualizaciones (heatmap, boxplots, ejemplos)
  - Guardar resultados en JSON y PNG
- [x] Ejecutar el script y verificar resultados

# TODO: Análisis de Intensidad de Píxeles en MNIST

- [x] Implementar `experiments/analyze_intensity.py`:
  - Cargar MNIST (train + test)
  - Calcular intensidad total por imagen (suma de píxeles)
  - Generar estadísticas globales y por dígito
  - Crear visualizaciones (histograma, boxplots, densidad, ejemplos)
  - Analizar correlación intensidad vs traslación
  - Guardar resultados en JSON y PNG
- [x] Ejecutar el script y verificar resultados

# TODO: Análisis de Firmas de Islas en MNIST

- [x] Implementar `experiments/analyze_island_signatures.py`:
  - Cargar MNIST (train + test)
  - Calcular firmas horizontales (islas por fila)
  - Calcular firmas verticales (islas por columna)
  - Generar estadísticas por dígito
  - Crear visualizaciones (heatmaps, líneas, distribuciones, ejemplos)
  - Guardar resultados en JSON y PNG
- [x] Ejecutar el script y verificar resultados

# TODO: Clasificador PAC con Firmas de Islas

- [x] Implementar `experiments/pac_island_signature_classifier.py`:
  - Cargar MNIST
  - Computar firmas de islas (56D) para train y test
  - Entrenar PAC-V2 con firmas de islas
  - Entrenar PAC-V2 con píxeles brutos (baseline)
  - Comparar accuracy, tiempo, archetypes
  - Generar matrices de confusión y visualizaciones
- [x] Ejecutar el script y verificar resultados
  - Accuracy firmas: 85.30%
  - Accuracy píxeles: 96.07%
  - Speedup: 1.7x
  - Reducción dimensión: 14x
