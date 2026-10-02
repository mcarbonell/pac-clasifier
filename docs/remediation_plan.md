# Plan de Remediación y Preparación para Publicación — PAC Classifier

Este plan operacionaliza las recomendaciones de la auditoría (`docs/audit_and_remediation.md`), respetando las normas de rigor científico, logging y hardware del repositorio (`GEMINI.md`).

Cada fase concluye con verificación, **commit** y **push** al repositorio remoto.

---

## 📋 Checklist General por Fases

- [x] **Fase 0: Higiene del Repositorio y Refactorización del Core**
- [ ] **Fase 1: Suite de Tests Automatizados (`pytest`)**
- [ ] **Fase 2: Benchmarks Multi-Dataset y Baselines Competitivos (LVQ, KNN, SVM, MLP)**
- [ ] **Fase 3: Experimentos de Robustez, Ruido Controlado y Ablaciones**
- [ ] **Fase 4: Formalización Matemática, Algoritmo y Documentación**
- [ ] **Fase 5: Estructuración y Redacción del Borrador del Paper**

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

- [ ] **1.1 Infraestructura de Tests:**
  - Crear directorio `tests/` y `conftest.py`.
- [ ] **1.2 Batería de Pruebas Unitarias:**
  - `test_fit_predict_synthetic`: Datos 2D sintéticos linealmente separables convergen a 100% de precisión.
  - `test_arbitrary_dimensions`: Funciona correctamente con dimensiones $D=1, 5, 50, 100$.
  - `test_archetype_monotonicity`: El número de arquetipos activos $K$ no decrece por generación.
  - `test_confusion_mapping`: Verificación de que en `cluster_confusion_map` los pares son siempre `true != pred`.
  - `test_persistent_errors`: Detección de `never_correct_mask` consistente.
  - `test_retrocompatibility`: Verificación de que `PurifyingArchetypeClassifierV2` desde `pac_v2` replica idéntico comportamiento.
  - `test_device_cpu`: Verificación de funcionamiento estricto en CPU sin dependencias GPU.
- [ ] **1.3 Ejecución y Aprobación:**
  - Correr `pytest` y confirmar que todos los tests pasan al 100%.
- [ ] **1.4 Commit & Push Fase 1.**

---

### Fase 2: Benchmarks Multi-Dataset y Baselines Competitivos
*Objetivo:* Generar la evidencia empírica rigurosa que exige la comunidad académica, comparando PAC contra baselines pertinentes.

- [ ] **2.1 Framework de Métricas (`GEMINI.md`):**
  - Implementar logger estandarizado que registre en `results/raw/`:
    - `wall_clock_time`, `function_evaluation_time`, `internal_overhead_time`.
    - `final_objective` (accuracy), `total_evaluations`, `convergence_speed`.
    - `num_seeds`, `std_objective`, `hardware_info`, `full_config`.
  - Incluir flag `--quick` / `--dry-run` para verificación inmediata en segundos y modo completo multi-seed para ejecución terminal.
- [ ] **2.2 Baseline LVQ en PyTorch:**
  - Implementar clasificador GLVQ (Generalized Learning Vector Quantization) vectorizado en PyTorch como baseline directo y justo frente a PAC.
- [ ] **2.3 Baselines Clásicos (Scikit-Learn / PyTorch):**
  - KNN ($k=1, 3, 5$).
  - SVM (RBF kernel).
  - MLP simple (1 capa oculta).
- [ ] **2.4 Evaluación Multi-Dataset (Multi-Seed):**
  - **Fashion-MNIST:** 784D, 10 clases (drop-in para sustituir la exclusividad de MNIST).
  - **Dataset Tabular:** (ej. Pendigits o Covertype reducido) para demostrar funcionamiento fuera del dominio de imágenes.
  - **MNIST Multi-Seed:** 5 a 10 semillas para reportar media $\pm$ desviación estándar.
- [ ] **2.5 Commit & Push Fase 2.**

---

### Fase 3: Experimentos de Robustez, Ruido Controlado y Ablaciones
*Objetivo:* Blindar la contribución de PAC como herramienta de auditoría de datasets y cartografía de fronteras.

- [ ] **3.1 Inyección de Ruido de Etiquetas Controlado:**
  - Inyectar 0%, 5%, 10%, 15%, 20% de etiquetas aleatorias simétricas en MNIST / Fashion-MNIST.
  - Evaluar la correlación entre la tasa de ruido inyectada y los `persistent_error_indices` de PAC.
- [ ] **3.2 Estudio de Ablación:**
  - Métrica de distancia: Similitud Coseno vs Distancia Euclídea ($L_2$).
  - Parámetro `min_cluster_size`: impacto en el número total de arquetipos vs precisión.
  - Curva de crecimiento $K(t)$ y coste temporal por generación.
- [ ] **3.3 Commit & Push Fase 3.**

---

### Fase 4: Formalización Matemática y Documentación
*Objetivo:* Dotar al algoritmo del lenguaje y formalismo de conferencias de primer nivel (NeurIPS/ICML/AISTATS).

- [ ] **4.1 Formalización Matemática (`docs/mathematical_formulation.md`):**
  - Notación formal de conjuntos de arquetipos $\mathcal{A} = \{(a_k, y_k)\}$.
  - Definición rigurosa de las dos fases por iteración: *Purification* (aislamiento/reasignación intra-clase) y *Bifurcation* (generación de arquetipos por matriz de confusión dirigida).
  - Demostración / análisis de no-migración inter-clase de aciertos.
  - Análisis formal de complejidad asintótica temporal $O(G \cdot N \cdot K_{avg} \cdot D)$ y espacial.
- [ ] **4.2 Pseudocódigo Formal:**
  - Estructuración en estilo Algorithm 1 con entradas, inicialización, bucle principal y salida.
- [ ] **4.3 Actualización de README.md:**
  - Reemplazar comparativas débiles por el nuevo posicionamiento estratégico, citando formalmente LVQ, Prototype Networks y Cleanlab.
- [ ] **4.4 Commit & Push Fase 4.**

---

### Fase 5: Estructuración y Redacción del Borrador del Paper
*Objetivo:* Dejar el manuscrito listo en formato académico.

- [ ] **5.1 Estructura del Manuscrito (`docs/paper/`):**
  - Abstract, Introduction, Related Work (LVQ, ProtoPNet, Dataset Cartography, Cleanlab).
  - Method (Mathematical Formulation + Algorithm 1).
  - Experiments & Results (Tablas comparativas, multi-seed, ablaciones, gráficos $K(t)$).
  - Dataset Auditing via Persistent Errors (Ruido controlado + validación en MNIST).
  - Limitations & Future Work.
- [ ] **5.2 Commit & Push Fase 5.**
