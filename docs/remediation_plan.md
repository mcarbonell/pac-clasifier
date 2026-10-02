# Plan de Remediación y Preparación para Publicación — PAC Classifier

Este plan operacionaliza las recomendaciones de la auditoría (`docs/audit_and_remediation.md`), respetando las normas de rigor científico, logging y hardware del repositorio (`GEMINI.md`).

Cada fase concluye con verificación, **commit** y **push** al repositorio remoto.

---

## 📋 Checklist General por Fases

- [x] **Fase 0: Higiene del Repositorio y Refactorización del Core**
- [x] **Fase 1: Suite de Tests Automatizados (`pytest`)**
- [x] **Fase 2: Benchmarks Multi-Dataset y Baselines Competitivos (LVQ, KNN, SVM, MLP)**
- [x] **Fase 3: Experimentos de Robustez, Ruido Controlado y Ablaciones**
- [x] **Fase 4: Formalización Matemática, Algoritmo y Documentación**
- [x] **Fase 5: Estructuración y Redacción del Borrador del Paper**

---

## Detalle de Fases y Tareas

### Fase 0: Higiene del Repositorio y Refactorización del Core
*Objetivo:* Dejar el repositorio limpio, profesional, instalable y con una arquitectura de código unificada y desacoplada sin romper compatibilidad histórica.

- [x] **0.1 Higiene de Git e Ignorados:**
  - Actualizar `.gitignore` para ignorar: `__pycache__/`, `*.pyc`, `data/`, `results/`, `*.egg-info/`, `.pytest_cache/`, `.coverage`, etc.
  - Mover el prototipo suelto en raíz `prototype_v78_pac_bifurcation.py` a `scratch/`.
- [x] **0.2 Corrección de Documentación Existente:**
  - Corregir el doble escape en fórmulas LaTeX (`\\\\frac` → `\frac`, `\\\\mu` → `\mu`, etc.) en `docs/findings.md`.
- [x] **0.3 Licencia y Empaquetado:**
  - Crear `LICENSE` (Licencia MIT a nombre de Mario Raúl Carbonell Martínez).
  - Crear `pyproject.toml` moderno (PEP 621 / setuptools) y actualizar `requirements.txt` con dependencias mínimas y de desarrollo (`pytest`, etc.).
- [x] **0.4 Refactorización y Unificación del Motor PAC (`pac/`):**
  - Unificar la lógica en `pac/classifier.py`:
    - Clase `PurifyingArchetypeClassifier` con parámetro `bifurcation_mode='confusion'` (V2 por defecto) y soporte para `bifurcation_mode='confidence'` (V1 legacy).
    - Hacer el clasificador **completamente agnóstico a la dimensión** (acepta tensores $(N, D)$ para cualquier $D$, no hardcodeado a $28 \times 28$).
    - Soporte limpio y robusto para `device` (`cpu`, `directml`, `cuda`).
    - Desacoplar el morphing espacial afín a un módulo auxiliar opcional (`pac/morphing.py`).
  - Crear shim/wrapper retrocompatible en `pac_v2/` (`pac_v2/__init__.py` y `pac_v2/classifier.py`) para que los scripts existentes que hacen `from pac_v2 import PurifyingArchetypeClassifierV2` sigan funcionando exactamente igual.
- [x] **0.5 Verificación de no-regresión:**
  - Probar importaciones y un ciclo rápido de validación.
- [x] **0.6 Commit & Push Fase 0.**

---

### Fase 1: Suite de Tests Automatizados
*Objetivo:* Garantizar la robustez del algoritmo con integración continua local mediante `pytest`.

- [x] **1.1 Infraestructura de Tests:**
  - Crear directorio `tests/` y `conftest.py`.
- [x] **1.2 Batería de Pruebas Unitarias:**
  - `test_fit_predict_synthetic`: Datos 2D sintéticos linealmente separables convergen a 100% de precisión.
  - `test_arbitrary_dimensions`: Funciona correctamente con dimensiones $D=1, 5, 50, 100$.
  - `test_archetype_monotonicity`: El número de arquetipos activos $K$ no decrece por generación.
  - `test_confusion_mapping`: Verificación de que en `cluster_confusion_map` los pares son siempre `true != pred`.
  - `test_persistent_errors`: Detección de `never_correct_mask` consistente.
  - `test_retrocompatibility`: Verificación de que `PurifyingArchetypeClassifierV2` desde `pac_v2` replica idéntico comportamiento.
  - `test_device_cpu`: Verificación de funcionamiento estricto en CPU sin dependencias GPU.
- [x] **1.3 Ejecución y Aprobación:**
  - Correr `pytest` y confirmar que todos los tests pasan al 100% (16 tests, 95% cobertura).
- [x] **1.4 Commit & Push Fase 1.**

---

### Fase 2: Benchmarks Multi-Dataset y Baselines Competitivos
*Objetivo:* Generar la evidencia empírica rigurosa que exige la comunidad académica, comparando PAC contra baselines pertinentes.

- [x] **2.1 Framework de Métricas (`GEMINI.md`):**
  - Implementado `experiments/metrics_logger.py` que registra estrictamente en `results/raw/` y `results/summary/`:
    - `wall_clock_time`, `function_evaluation_time`, `internal_overhead_time`.
    - `final_objective` (accuracy), `total_evaluations`, `convergence_speed`.
    - `num_seeds`, `std_objective`, `hardware_info`, `full_config`.
  - Soporte `--quick` para Fast Feedback y verificación instantánea en segundos.
- [x] **2.2 Baseline LVQ en PyTorch:**
  - Implementado clasificador GLVQ (Generalized Learning Vector Quantization - Sato & Yamada 1996) vectorizado en PyTorch en `experiments/baselines/glvq.py`.
- [x] **2.3 Baselines Clásicos (Scikit-Learn):**
  - KNN ($k=3$), SVM (RBF kernel), y MLP (1 capa oculta) unificados en `experiments/baselines/classical.py`.
- [x] **2.4 Suite y Datasets:**
  - `experiments/datasets.py` para MNIST, Fashion-MNIST y Tabular (Digits).
  - Runner comparativo unificado `experiments/run_benchmarks.py` con validación `--quick` verificada al 100%.
- [x] **2.5 Commit & Push Fase 2.**

---

### Fase 3: Experimentos de Robustez, Ruido Controlado y Ablaciones
*Objetivo:* Blindar la contribución de PAC como herramienta de auditoría de datasets y cartografía de fronteras.

- [x] **3.1 Inyección de Ruido de Etiquetas Controlado:**
  - Implementado `experiments/analyze_robustness_noise.py` con inyección de 0%, 5%, 10%, 15%, 20% de ruido aleatorio simétrico.
  - Medida y confirmada la correlación monótona entre ruido inyectado y tasa de errores persistentes detectada.
  - Gráfica generada en `results/figures/tabular_noise_injection_analysis.png` y resumen en `results/summary/tabular_noise_injection_summary.json`.
- [x] **3.2 Estudio de Ablación y Dinámica $K(t)$:**
  - Implementado `experiments/analyze_ablations.py`.
  - Ablación de métrica de distancia: Similitud Coseno (97.78%) vs Distancia Euclídea (96.39%).
  - Ablación del parámetro `min_cluster_size` (1 a 20) demostrando trade-off óptimo capacidad vs precisión.
  - Curvas de trayectoria $K(t)$ y coste acumulado por generación.
  - Gráficas generadas en `results/figures/` y resúmenes en `results/summary/`.
- [x] **3.3 Commit & Push Fase 3.**

---

### Fase 4: Formalización Matemática y Documentación
*Objetivo:* Dotar al algoritmo del lenguaje y formalismo de conferencias de primer nivel (NeurIPS/ICML/AISTATS).

- [x] **4.1 Formalización Matemática (`docs/mathematical_formulation.md`):**
  - Notación formal de conjuntos de arquetipos $\mathcal{A} = \{(a_k, y_k)\}$.
  - Definición rigurosa de las dos fases por iteración: *Purification* (aislamiento/reasignación intra-clase) y *Bifurcation* (generación de arquetipos por matriz de confusión dirigida).
  - Demostración formal (Proposición 1) de no-migración inter-clase de aciertos.
  - Análisis formal de complejidad asintótica temporal $\mathcal{O}(G \cdot N \cdot \bar{K} \cdot D)$ y espacial.
- [x] **4.2 Pseudocódigo Formal:**
  - Estructuración en estilo Algorithm 1 con entradas, inicialización, bucle principal y salida.
- [x] **4.3 Actualización de README.md:**
  - Reemplazadas comparativas informales por el nuevo posicionamiento estratégico, citando formalmente LVQ, Prototype Networks y Cleanlab, incorporando tablas empíricas reales de benchmarks (Tabular, Fashion-MNIST, MNIST) y ejemplos de uso.
- [x] **4.4 Commit & Push Fase 4.**

---

### Fase 5: Estructuración y Redacción del Borrador del Paper
*Objetivo:* Dejar el manuscrito listo en formato académico.

- [x] **5.1 Estructura del Manuscrito (`docs/paper/paper_draft.md`):**
  - Abstract, Introduction y posicionamiento estratégico.
  - Related Work exhaustivo (LVQ/GLVQ, ProtoNets, ProtoPNet, Dataset Cartography, Confident Learning / Cleanlab).
  - Method (Definiciones, Dual-phase dynamics, Proposición 1 de no-deriva inter-clase, Algorithm 1 y análisis de complejidad).
  - Experiments & Results (Tablas comparativas en Tabular, Fashion-MNIST y MNIST vs GLVQ, KNN, SVM, MLP).
  - Dataset Auditing via Persistent Errors (Ruido controlado 0%-20% y validación con los 261 errores de MNIST).
  - Ablations (Coseno vs Euclídeo, `min_cluster_size` y trayectoria $K(t)$).
  - Discussion & Limitations.
- [x] **5.2 Commit & Push Fase 5.**
