# Auditoría Completa — PAC Classifier

**Autor de la auditoría:** Antigravity (Claude Opus 4.6)  
**Fecha:** 2 de octubre de 2026  
**Repositorio:** `mcarbonell/pac-classifier`  
**Objetivo:** Evaluar la viabilidad de publicación académica y definir un plan de remediación.

---

## Índice

1. [Resumen Ejecutivo](#1-resumen-ejecutivo)
2. [Auditoría de la Idea (Novelty & Positioning)](#2-auditoría-de-la-idea)
3. [Auditoría del Código](#3-auditoría-del-código)
4. [Auditoría de la Documentación](#4-auditoría-de-la-documentación)
5. [Auditoría de la Evidencia Empírica](#5-auditoría-de-la-evidencia-empírica)
6. [Análisis de Debilidades Críticas](#6-análisis-de-debilidades-críticas)
7. [Plan de Remediación](#7-plan-de-remediación)

---

## 1. Resumen Ejecutivo

PAC (Purifying Archetype Classifier) es un algoritmo de clasificación supervisada basado en prototipos que construye dinámicamente centroides ("arquetipos") mediante aislamiento iterativo de errores. La versión V2 introduce bifurcación semántica por tipo de confusión (`true→predicted`), creando una cartografía interpretable de los límites de decisión entre clases.

### Veredicto General

| Dimensión | Estado | Notas |
|-----------|--------|-------|
| **Novelty de la idea** | 🟢 Sólida | Combinación original de prototype learning + error-driven bifurcation |
| **Positioning académico** | 🟡 Necesita reframing | Se compara con K-Means/KNN (demasiado débiles); falta comparar con Prototype Networks, ProtoPNet, LVQ |
| **Calidad del código** | 🟡 Funcional pero inmaduro | Sin tests, sin validación cruzada, hardcoded a MNIST 28×28, dos paquetes paralelos sin unificar |
| **Evidencia empírica** | 🔴 Insuficiente para paper | Solo MNIST, sin multi-seed, sin baselines competitivos, sin otros datasets |
| **Documentación** | 🟡 Buena para repo, débil para paper | README excelente; falta formalización matemática, pseudocódigo riguroso, análisis de complejidad |
| **Reproducibilidad** | 🔴 Frágil | Sin requirements.txt funcional, sin seeds fijadas, rutas relativas inconsistentes |

---

## 2. Auditoría de la Idea

### 2.1 Contribuciones Originales Identificadas

1. **Error-Driven Prototype Spawning**: A diferencia de K-Means (que usa distorsión) o LVQ (que usa push/pull de prototipos existentes), PAC *crea nuevos prototipos donde hay errores*. Esta es una contribución legítima.

2. **Confusion-Aware Bifurcation (V2)**: Agrupar errores por `(true_label, predicted_label)` y crear un arquetipo específico para esa confusión es, hasta donde puedo verificar, una idea original. Cada arquetipo tiene una semántica interpretable: "los 4 que parecen 9".

3. **Dataset Auditing Emergente**: El uso de errores persistentes como detector de mislabels, con resultados que convergen con cleanlab (0.43% vs 0.44%), es un resultado empírico potente.

### 2.2 Trabajo Relacionado NO Citado (Crítico)

| Área | Trabajo | Relevancia |
|------|---------|------------|
| **Prototype Networks** | Snell et al. (NeurIPS 2017) | Clasificación por distancia a prototipos aprendidos. Diferencia clave: PAC usa medias, no embeddings aprendidos. |
| **Learning Vector Quantization (LVQ)** | Kohonen (1990); GLVQ, RSLVQ | Familia completa de clasificadores por prototipos con reglas de actualización supervisadas. **El ancestro más directo de PAC.** |
| **ProtoPNet** | Chen et al. (NeurIPS 2019) | Prototipos interpretables para deep learning. Similar filosofía de "este se parece a aquello". |
| **Confident Learning** | Northcutt et al. (JAIR 2021) | El baseline explícito para la claim de dataset auditing. |
| **Growing Neural Gas / ESOM** | Fritzke (1995) | Redes de prototipos que crecen dinámicamente, similar a cómo PAC hace crecer K. |
| **Hierarchical Prototype Networks** | Varios | Linaje de prototipos con estructura jerárquica. |

> ⚠️ **ATENCIÓN: La ausencia de comparación con LVQ es la debilidad más grave para publicación.** Un revisor experto en prototype learning rechazará el paper inmediatamente si no se menciona LVQ y se explica por qué PAC es diferente/mejor. La diferencia clave es: LVQ *mueve* prototipos existentes; PAC *crea* nuevos prototipos y *purifica* los existentes por aislamiento.

### 2.3 Framing Recomendado

El feedback en `docs/private/feedback.md` ya lo sugiere, y estoy de acuerdo: **no posicionar como competidor de clasificadores generales** (ahí pierde contra cualquier red neuronal trivial). Posicionar como:

> **"Supervised error-driven prototype spawning for interpretable classification and dataset cartography"**

Esto lo alinea con:
- Prototype-based interpretable ML (ProtoPNet, This Looks Like That)
- Dataset cartography (Swayamdipta et al., EMNLP 2020)
- Label error detection (Northcutt/cleanlab)

---

## 3. Auditoría del Código

### 3.1 Arquitectura del Código

```
pac/              ← V1: bifurcación por mediana de confianza (128 líneas)
pac_v2/           ← V2: bifurcación por confusión (356 líneas)
prototype_v78_... ← Script standalone (raíz, no empaquetado)
examples/         ← 6 scripts de ejemplo
experiments/      ← 4 scripts de análisis
```

### 3.2 Problemas Identificados

#### P1: Dos paquetes paralelos sin unificar (Severidad: Media)

`pac/` y `pac_v2/` son módulos independientes con código duplicado (~80% del bucle de entrenamiento es idéntico). Para publicación, debe haber **un solo módulo** con la versión canónica (V2), y V1 puede ser un caso particular (`bifurcation_mode='confidence'` vs `'confusion'`).

#### P2: Hardcoded a 28×28 (Severidad: Alta)

En `pac_v2/classifier.py` L194, el método `_morph_archetype` tiene:
```python
img = archetype.view(1, 1, 28, 28)  # ← hardcoded
grid = F.affine_grid(matrix, [1, 1, 28, 28], ...)  # ← hardcoded
```
Esto impide usar PAC con cualquier dataset que no sea MNIST 28×28. Para publicación necesita aceptar dimensiones arbitrarias.

#### P3: `import Adam` no utilizado en el flujo principal (Severidad: Baja)

En `pac_v2/classifier.py` L3:
```python
from torch.optim import Adam  # Se usa dentro de _optimize_fit
```
Se usa en `_optimize_fit`, pero este método solo se invoca desde `predict_with_morphing`, que es una feature experimental. No es un bug, pero el morphing debería estar documentado como experimental.

#### P4: Sin tests unitarios (Severidad: Crítica)

No existe carpeta `tests/`. No hay ni un solo test automatizado. Para publicación se necesita al mínimo:
- Test de que `fit()` + `predict()` funcionan end-to-end en datos sintéticos
- Test de que el número de arquetipos crece monotónicamente
- Test de que `cluster_history` es consistente
- Test de convergencia a 100% en datos trivialmente separables

#### P5: Sin seed fijada (Severidad: Alta)

Ningún script fija `torch.manual_seed()` ni `random.seed()`. Los resultados no son reproducibles. Hay que fijar seeds en todos los experiments y reportar media ± std sobre múltiples seeds.

#### P6: `device` hardcodeado a CUDA (Severidad: Media)

En `pac/classifier.py` L13 y `pac_v2/classifier.py` L23:
```python
self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```
Esto está correcto pero los scripts standalone como `prototype_v78` y varios experiments también hacen `'cuda' if torch.cuda.is_available()` sin considerar DirectML. Según las reglas del repo, tu máquina no tiene CUDA.

#### P7: `prototype_v78_pac_bifurcation.py` en la raíz (Severidad: Baja)

Archivo suelto en raíz del proyecto. Parece ser un prototipo histórico de V1. Debería estar en `scratch/` o eliminarse.

#### P8: Rutas relativas inconsistentes (Severidad: Media)

Los examples usan `'../data'` para MNIST, los experiments también `'../data'`, pero el prototype_v78 usa `'./data'`. Esto hace que los scripts solo funcionen si se ejecutan desde su directorio exacto. Necesita una constante global o path relativo al proyecto root.

#### P9: Bug potencial en reasignación de correctos (Severidad: Media-Alta)

En el paso de purificación (Paso 2 del bucle), un sample correctamente clasificado se reasigna al arquetipo MÁS CERCANO:
```python
image_cluster_assignment[correct] = arch_cluster_ids[best_arch_idx[correct]]
```
Si un '4' correctamente clasificado tiene como arquetipo más cercano un sub-arquetipo con label '4' pero perteneciente a una confusión `4→9`, la reasignación es correcta en label pero semánticamente confusa. **Esto no es un bug funcional** (el label del arquetipo sigue siendo '4'), pero puede hacer que un '4' limpio migre al cluster de "4s que parecen 9s", contaminando ese arquetipo. El feedback en `docs/private/feedback.md` (punto 1) señala exactamente esta ambigüedad. **Debe documentarse o resolverse.**

#### P10: Escalabilidad O(G·N·K·D) con K creciente (Severidad: Media)

Cada generación recalcula `cos_sim = torch.mm(norm_train, norm_arch.t())`, que es O(N·K·D). Como K crece en cada iteración (de 10 a ~1470 en 100 generaciones), el coste total no es lineal en G. El paper debe incluir una curva de K(t) y coste acumulado real, como sugiere el feedback.

### 3.3 Calidad del Código: Resumen

| Aspecto | Puntuación | Comentario |
|---------|:----------:|------------|
| Legibilidad | 8/10 | Código limpio, bien comentado, buen naming |
| Modularidad | 5/10 | Duplicación V1/V2, morphing acoplado al clasificador |
| Robustez | 4/10 | Sin tests, sin validación de inputs, sin seeds |
| Portabilidad | 3/10 | Hardcoded a MNIST 28×28, paths inconsistentes |
| Documentación inline | 7/10 | Docstrings presentes, comentarios explicativos buenos |

---

## 4. Auditoría de la Documentación

### 4.1 README.md

**Fortalezas:**
- Explica el algoritmo de forma clara y accesible
- Buenas tablas comparativas (PAC vs K-Means vs KNN)
- Resultados empíricos bien presentados (auditoría MNIST)
- Imágenes de referencia

**Debilidades:**
- Falta notación matemática formal (el paper necesitará definiciones de A, f_label, la métrica de distancia, etc.)
- No menciona LVQ ni trabajo relacionado académico
- El `>99%` de eficiencia sobre KNN es engañoso: KNN con ball trees no requiere almacenar 60K vectores en bruto, hay implementaciones eficientes
- Falta sección de limitaciones
- Falta mención de licencia

### 4.2 findings.md

**Fortalezas:**
- Excelente documento de laboratorio: metodología → resultados → análisis → conclusiones
- 4 experimentos bien documentados
- Tablas claras y completas

**Debilidades:**
- Fórmulas LaTeX con doble escape (`\\\\frac` en lugar de `\frac`): se renderizan mal
- Solo cubre los análisis exploratorios (traslación, intensidad, firmas de islas), no el algoritmo PAC en sí
- No hay comparación formal con baselines

### 4.3 Documentación Faltante para Paper

- [ ] Pseudocódigo formal del algoritmo (estilo Algorithm 1 en LaTeX)
- [ ] Definiciones matemáticas rigurosas
- [ ] Análisis de convergencia (¿converge siempre? ¿bajo qué condiciones?)
- [ ] Análisis de complejidad temporal y espacial con K(t)
- [ ] Sección de trabajo relacionado
- [ ] Sección de limitaciones

---

## 5. Auditoría de la Evidencia Empírica

### 5.1 Resultados Actuales

| Métrica | Valor Reportado | Suficiente para Paper? |
|---------|----------------|:----------------------:|
| MNIST Test Accuracy (V2, pixels) | 96.07% | 🔴 No (sin baseline comparativo formal) |
| Arquetipos descubiertos | 1,470 | ✅ Dato interesante |
| Errores persistentes MNIST | 261 (0.43%) | 🟢 Fuerte claim si se replica |
| Coincidencia con cleanlab | ~0.44% | 🟢 Potente |
| Island Signatures accuracy | 85.30% | ✅ Interesante como ablation |

### 5.2 Debilidades Empíricas Críticas

#### E1: Solo MNIST (Severidad: Crítica)

Ningún venue serio aceptará un paper de clasificación evaluado SOLO en MNIST. Se necesitan al mínimo:
- **Fashion-MNIST** (mismas dimensiones, plug-and-play)
- **CIFAR-10** (requiere flatten a 3072D o embeddings)
- Un dataset tabular (para demostrar generalidad fuera de imágenes)

#### E2: Sin baselines competitivos (Severidad: Crítica)

Se compara con K-Means y KNN conceptualmente, pero no hay una tabla formal como:

| Método | MNIST Acc | Fashion-MNIST Acc | #Prototipos | Tiempo |
|--------|-----------|-------------------|-------------|--------|
| KNN (k=3) | ~97.1% | — | 60,000 | — |
| LVQ-2.1 | ~96% | — | ~100 | — |
| SVM (RBF) | ~98.5% | — | — | — |
| **PAC-V2** | **96.07%** | — | **1,470** | — |

#### E3: Sin multi-seed (Severidad: Alta)

Todos los resultados son de una sola ejecución. Sin `mean ± std` sobre 5+ seeds, los resultados no son estadísticamente fiables.

#### E4: Sin experimento de ruido controlado (Severidad: Media)

El feedback ya lo sugiere: inyectar 5%, 10%, 15% de labels aleatorios y verificar que PAC detecta tasas cercanas a las inyectadas. Esto convertiría la coincidencia MNIST/cleanlab de "interesante" a "riguroso".

#### E5: Sin análisis de sensibilidad a hiperparámetros (Severidad: Media)

PAC tiene pocos hiperparámetros (`max_iters`, `target_acc`, `min_cluster_size`), lo cual es una fortaleza. Pero falta un estudio de ablación que muestre cómo varían los resultados al cambiar estos valores.

---

## 6. Análisis de Debilidades Críticas

### 6.1 La Pregunta del Revisor Inevitable: "¿Cómo se diferencia de LVQ?"

**Learning Vector Quantization (LVQ)** es una familia de algoritmos supervisados que mantienen prototipos y los actualizan basándose en si la clasificación es correcta o incorrecta. Un revisor familiarizado con LVQ preguntará:

| Aspecto | LVQ | PAC |
|---------|-----|-----|
| Número de prototipos | Fijo | Dinámico (crece) |
| Actualización de prototipos | Gradiente (push/pull) | Re-promediado completo |
| Creación de prototipos | No | Sí (por aislamiento de errores) |
| Interpretabilidad del linaje | No | Sí (confusion-aware) |
| Complejidad por iteración | O(N·K·D) | O(N·K·D) |

**Respuesta preparada**: PAC no optimiza prototipos existentes — los *purifica* por exclusión. Esto es fundamentalmente diferente de LVQ: en lugar de mover un prototipo hacia los ejemplos correctos (y lejos de los incorrectos), PAC *elimina los incorrectos del cálculo del prototipo* y les da su propio prototipo. Esto produce prototipos que son medias "limpias" de subconjuntos homogéneos, no compromisos entre gradientes opuestos.

### 6.2 ¿Converge siempre?

No está demostrado. Empíricamente se estanca en ~97% en MNIST. El README dice "Repeat until accuracy reaches 100% or max generations", pero no hay demostración de que:
- El accuracy sea monotónicamente creciente
- El algoritmo termine en tiempo finito sin el cap de `max_iters`
- No haya oscilaciones (un sample que alterna entre correcto e incorrecto)

> **IMPORTANTE:** Un análisis teórico (o al menos empírico extenso) de convergencia es necesario para el paper. Al mínimo, mostrar curvas de accuracy(t) y K(t) para múltiples datasets y seeds.

### 6.3 Sensibilidad a la Métrica de Distancia

PAC usa similitud coseno. ¿Funciona con L2? ¿Con distancia de Mahalanobis? Un ablation sobre la métrica de distancia fortalecería la contribución.

---

## 7. Plan de Remediación

### Fase 0: Higiene del Repositorio (1-2 días)

| Tarea | Prioridad | Archivo(s) |
|-------|-----------|------------|
| Mover `prototype_v78_pac_bifurcation.py` a `scratch/` | Baja | Raíz → `scratch/` |
| Unificar `pac/` y `pac_v2/` en un solo módulo `pac/` con parámetro `bifurcation_mode` | Alta | `pac/classifier.py` |
| Eliminar hardcoding 28×28 del morphing | Media | `pac_v2/classifier.py` |
| Añadir `requirements.txt` o migrar a `pyproject.toml` | Alta | Raíz |
| Corregir `.gitignore` (añadir `__pycache__/`, `*.pyc`, `data/MNIST/`, `results/`, `*.egg-info/`) | Media | `.gitignore` |
| Fijar seeds en todos los scripts | Alta | `examples/`, `experiments/` |
| Normalizar rutas (usar path relativo al proyecto root) | Media | Todos los scripts |
| Añadir `LICENSE` | Alta | Raíz |

### Fase 1: Tests Unitarios (1-2 días)

| Test | Qué verifica |
|------|-------------|
| `test_fit_predict_synthetic` | Datos 2D trivialmente separables → 100% accuracy |
| `test_archetype_count_grows` | K monótonamente no-decreciente por generación |
| `test_cluster_history_consistent` | Cada cluster en `arch_cluster_ids` tiene entrada en `cluster_history` |
| `test_confusion_map_correct` | Los pares `(true, pred)` en `cluster_confusion_map` nunca tienen `true == pred` |
| `test_persistent_errors_subset` | `persistent_error_indices` ⊂ {0, ..., N-1} |
| `test_predict_shape` | Output de `predict()` tiene shape (N,) para labels y (N,) para scores |
| `test_device_agnostic` | Funciona en CPU sin errores |

### Fase 2: Baselines y Multi-Dataset (3-5 días)

> **IMPORTANTE:** Esta es la fase más crítica para la viabilidad del paper.

| Experimento | Prioridad | Descripción |
|-------------|-----------|-------------|
| **Multi-seed MNIST** | Crítica | 10 seeds, reportar media ± std de accuracy, #archetypes, tiempo |
| **Fashion-MNIST** | Crítica | Mismo pipeline, mismos hiperparámetros. Plug-and-play (28×28 grayscale) |
| **CIFAR-10 flattened** | Alta | 32×32×3 = 3072D. Esperar accuracy baja pero demostrar generalidad |
| **Baseline KNN** | Crítica | sklearn KNeighborsClassifier(k=1,3,5,7) en los mismos datasets |
| **Baseline LVQ** | Crítica | sklearn-lvq o implementación propia, mismo K que PAC final |
| **Baseline SVM** | Alta | sklearn SVC(kernel='rbf') como upper bound de métodos clásicos |
| **Baseline MLP** | Media | 1-hidden-layer MLP (784→256→10) como referencia de deep learning mínimo |

### Fase 3: Experimentos de Robustez (2-3 días)

| Experimento | Prioridad | Descripción |
|-------------|-----------|-------------|
| **Ruido controlado** | Crítica | Inyectar 0%, 5%, 10%, 15%, 20% de labels aleatorios en MNIST. Medir: (a) accuracy final de PAC, (b) tasa de errores persistentes vs tasa inyectada. Graficar correlación. |
| **Ablation: métrica de distancia** | Alta | Coseno vs L2 vs L1. ¿Cuál da mejor accuracy/convergencia? |
| **Ablation: min_cluster_size** | Media | Variar de 1 a 50. ¿Cómo afecta a accuracy vs #archetypes? |
| **Curva K(t)** | Alta | Graficar número de arquetipos vs generación para todos los datasets |
| **Curva accuracy(t)** | Alta | Convergencia por generación, múltiples seeds |

### Fase 4: Formalización Matemática (2-3 días)

| Entregable | Descripción |
|------------|-------------|
| **Definiciones formales** | A = {(a_k, l_k)} el conjunto de arquetipos. C_k ⊆ X el cluster k. a_k = (1/\|C_k\|)·Σ x. |
| **Algorithm 1** | Pseudocódigo riguroso estilo NeurIPS/ICML |
| **Proposición de convergencia** | Al menos un análisis empírico: "accuracy es monotónicamente no-decreciente en todos los experimentos realizados" |
| **Análisis de complejidad** | Tiempo: O(G · N · K_avg · D). Espacio: O(K_max · D + N) |

### Fase 5: Paper Draft (3-5 días)

| Sección | Contenido Clave |
|---------|----------------|
| **Abstract** | "We present PAC, a non-differentiable prototype-based classifier that iteratively purifies archetypes by isolating classification errors..." |
| **Introduction** | Motivación: interpretabilidad + eficiencia. No competir con SOTA accuracy. |
| **Related Work** | LVQ, Prototype Networks, ProtoPNet, Growing Neural Gas, cleanlab |
| **Method** | Algorithm 1 + formalización matemática |
| **Experiments** | Multi-dataset, multi-seed, baselines, ablations |
| **Results: Classification** | Tabla comparativa con baselines |
| **Results: Dataset Auditing** | Experimento de ruido controlado + MNIST audit |
| **Discussion** | Limitaciones honestas, cuándo usar PAC vs DL |
| **Conclusion** | Contribución: error-driven prototype spawning + confusion-aware bifurcation |

### Fase 6: Cleanup Final (1 día)

| Tarea | Descripción |
|-------|-------------|
| Eliminar `pac/` (V1) si se unifica | Consolidar en un solo paquete |
| Mover scripts experimentales completados a `experiments/completed/` | Organizar |
| Verificar que `pip install -e .` funciona | Instalabilidad |
| Generar figuras finales de alta calidad | Para el paper |
| README bilingüe o solo en inglés | Para publicación internacional |

---

## Cronograma Estimado

| Fase | Duración | Dependencias |
|------|----------|-------------|
| Fase 0: Higiene | 1-2 días | — |
| Fase 1: Tests | 1-2 días | Fase 0 |
| Fase 2: Baselines & datasets | 3-5 días | Fase 1 |
| Fase 3: Robustez | 2-3 días | Fase 2 |
| Fase 4: Formalización | 2-3 días | Fase 2 (paralelo a F3) |
| Fase 5: Paper draft | 3-5 días | Fase 3 + Fase 4 |
| Fase 6: Cleanup | 1 día | Fase 5 |

**Tiempo total estimado:** 3-4 semanas de trabajo intermitente.

---

## Apéndice A: Inventario Completo de Archivos

| Archivo | Líneas | Propósito | Estado |
|---------|--------|-----------|--------|
| `pac/classifier.py` | 128 | V1: bifurcación por confianza | Funcional, candidato a deprecar |
| `pac_v2/classifier.py` | 356 | V2: bifurcación por confusión | Funcional, versión canónica |
| `prototype_v78_pac_bifurcation.py` | 146 | Prototipo histórico V1 standalone | Mover a scratch/ |
| `examples/mnist_example.py` | 42 | Demo V1 en MNIST | Funcional |
| `examples/mnist_pac_v2.py` | 96 | Demo V2 en MNIST | Funcional |
| `examples/audit_mnist_errors.py` | 170 | Auditoría de mislabels | Funcional, buen script |
| `examples/evaluate_morphing.py` | 62 | Evaluación de morphing (experimental) | Funcional, experimental |
| `examples/visualize_v1_archetypes.py` | 74 | Visualización arquetipos V1 | Funcional |
| `examples/visualize_v2_archetypes.py` | 107 | Visualización arquetipos V2 | Funcional |
| `experiments/analyze_translation.py` | 426 | Análisis de traslación MNIST | Completado |
| `experiments/analyze_intensity.py` | ~380 | Análisis de intensidad MNIST | Completado |
| `experiments/analyze_island_signatures.py` | ~430 | Firmas de islas MNIST | Completado |
| `experiments/pac_island_signature_classifier.py` | 379 | PAC con firmas de islas | Completado |
| `docs/findings.md` | 380 | Hallazgos de experimentos | Bueno, fórmulas LaTeX rotas |
| `README.md` | 119 | Documentación principal | Bueno para repo, insuficiente para paper |
| `setup.py` | 21 | Packaging | Migrar a pyproject.toml |

## Apéndice B: Respuestas Preparadas para el Feedback Recibido

El feedback en `docs/private/feedback.md` señala 3 dudas técnicas. Aquí las respuestas que el paper debe incluir:

**Duda 1: ¿Permite migración entre clases en la reasignación?**
→ **No.** Cuando un sample correctamente clasificado se reasigna, se reasigna al arquetipo más cercano *que ya tiene el mismo label*. Esto ocurre naturalmente porque la condición de "correctamente clasificado" implica que el arquetipo más cercano tiene el label correcto. Sin embargo, la reasignación es a un *sub-arquetipo* de esa clase, lo que permite migración intra-clase (ej: de "4s puros" a "4s que parecían 9s pero ahora son reconocidos"). **Esto debe explicitarse en el paper.**

**Duda 2: ¿Qué pasa con clases bimodales sin confusión?**
→ **PAC no las subdivide.** Esto es una limitación real: la resolución interna depende de confusión con otras clases. Debe discutirse como limitación y distinguirlo de K-Means, que sí puede dividir internamente por distorsión.

**Duda 3: Complejidad con K creciente.**
→ Incluir tabla/gráfica de K(t) y coste acumulado experimental. Ver Fase 4 del plan.

---

> **NOTA:** Este documento debe tratarse como punto de partida. Las prioridades pueden reordenarse según el venue objetivo (workshop vs conferencia main track vs journal).
