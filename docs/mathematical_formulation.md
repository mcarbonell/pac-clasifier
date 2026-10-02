# Mathematical Formulation of the Purifying Archetype Classifier (PAC)

**Author:** Mario Raúl Carbonell Martínez  
**Project:** `pac-classifier`  
**Target:** Academic publication in prototype learning and interpretable machine learning.

---

## 1. Problem Formulation & Notation

Let $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$ be a supervised training dataset, where:
- $x_i \in \mathcal{X} \subseteq \mathbb{R}^D$ denotes a $D$-dimensional feature vector.
- $y_i \in \mathcal{Y} = \{0, 1, \dots, C-1\}$ denotes the discrete class label.
- $N$ is the number of training instances, $D$ is the input dimensionality, and $C$ is the number of classes.

### Archetype Set
An **archetype** is defined as an annotated prototype pair:
$$a_k = (\mu_k, l_k) \in \mathbb{R}^D \times \mathcal{Y}$$
where $\mu_k \in \mathbb{R}^D$ is the geometric centroid vector and $l_k \in \mathcal{Y}$ is the associated categorical ground-truth class.

At any generation $t \ge 0$, the classifier maintains an active set of $K^{(t)}$ archetypes:
$$\mathcal{A}^{(t)} = \{(\mu_k^{(t)}, l_k^{(t)})\}_{k=1}^{K^{(t)}}$$

### Affinity Metric
For any query sample $x \in \mathbb{R}^D$ and archetype centroid $\mu_k \in \mathbb{R}^D$, the standard similarity function is the Cosine Affinity:
$$S(x, \mu_k) = \frac{\langle x, \mu_k \rangle}{\|x\|_2 \|\mu_k\|_2}$$

*(Alternatively, for unnormalized metric spaces, PAC supports negative squared Euclidean distance: $S(x, \mu_k) = -\|x - \mu_k\|_2^2$.)*

### Hypothesis Function
Given the active archetype set $\mathcal{A}^{(t)}$, the decision rule assigns $x$ the label of its nearest archetype:
$$k^*(x) = \arg\max_{k \in \{1, \dots, K^{(t)}\}} S(x, \mu_k^{(t)})$$
$$\hat{y}(x) = l_{k^*(x)}^{(t)}$$

---

## 2. Dynamic Archetype Evolution

The PAC algorithm operates over discrete generations $t = 0, 1, \dots, G-1$, transitioning partition assignments $\gamma_i^{(t)} \in \{1, \dots, K^{(t)}\}$ for each sample $i \in \{1, \dots, N\}$.

### Phase 1: Initialization ($t = 0$)
The algorithm initializes exactly one archetype per class ($K^{(0)} = C$):
$$\mathcal{C}_c^{(0)} = \{i \in \{1, \dots, N\} \mid y_i = c\}, \quad \forall c \in \mathcal{Y}$$
$$\mu_c^{(0)} = \frac{1}{|\mathcal{C}_c^{(0)}|} \sum_{i \in \mathcal{C}_c^{(0)}} x_i, \quad l_c^{(0)} = c$$
Each sample begins assigned to its class base archetype: $\gamma_i^{(0)} = y_i$.

---

### Phase 2: Iterative Purification Loop

At generation $t$, given the cluster assignment partition $\{\mathcal{C}_k^{(t)}\}_{k=1}^{K^{(t)}}$:

#### Step 1: Centroid Computation
Each active archetype's centroid is computed as the pure empirical mean of its currently assigned samples:
$$\mu_k^{(t)} = \frac{1}{|\mathcal{C}_k^{(t)}|} \sum_{i \in \mathcal{C}_k^{(t)}} x_i$$

#### Step 2: Evaluation & Classification
Evaluate the entire training set against $\mathcal{A}^{(t)}$:
$$k_i^* = \arg\max_{k \in \{1, \dots, K^{(t)}\}} S(x_i, \mu_k^{(t)})$$
$$\hat{y}_i^{(t)} = l_{k_i^*}^{(t)}$$

Let $\mathcal{E}^{(t)}$ denote the misclassified set and $\Omega^{(t)}$ denote the correctly classified set:
$$\Omega^{(t)} = \{i \in \{1, \dots, N\} \mid \hat{y}_i^{(t)} = y_i\}$$
$$\mathcal{E}^{(t)} = \{i \in \{1, \dots, N\} \mid \hat{y}_i^{(t)} \neq y_i\}$$

#### Step 3: Intra-Class Purification (Reassignment of Correct Samples)
For every correctly classified instance $i \in \Omega^{(t)}$, reassign it to its closest archetype:
$$\gamma_i^{(t+1)} = k_i^*$$

> **Proposition 1 (Strict Class Preservation of Correct Samples / No Inter-Class Drift):**  
> *For every $i \in \Omega^{(t)}$, the label of the assigned archetype strictly matches the true sample label:*
> $$l_{\gamma_i^{(t+1)}} = y_i$$
> **Proof:** By definition of $\Omega^{(t)}$, $\hat{y}_i^{(t)} = y_i$. Since the predicted label is defined by the nearest archetype $\hat{y}_i^{(t)} = l_{k_i^*}^{(t)}$, it follows immediately that $l_{k_i^*}^{(t)} = y_i$. Thus, reassignment $\gamma_i^{(t+1)} = k_i^*$ strictly maps sample $i$ to an archetype belonging to class $y_i$. There is **zero inter-class drift** among correct samples. $\blacksquare$

> **Remark on Intra-Class Migration:**  
> While inter-class migration is forbidden by Proposition 1, **intra-class migration** is explicitly permitted and encouraged: if class $c$ has spawned multiple sub-archetypes (e.g., pure 4s vs distorted 4s), a correct sample migrates to whichever sub-archetype best matches its localized sub-manifold, refining sub-cluster purity.

---

### Phase 3: Directed Confusion Bifurcation (Spawning)

Unlike classical LVQ (which pushes/pulls existing prototypes along a gradient) or K-Means (which partitions by spatial variance), PAC **spawns new archetypes exclusively from the structure of classification errors**.

For each true class $c_{true} \in \mathcal{Y}$ and each confused predicted class $c_{pred} \in \mathcal{Y} \setminus \{c_{true}\}$, we define the directed error set:
$$\mathcal{E}_{c_{true} \to c_{pred}}^{(t)} = \{i \in \mathcal{E}^{(t)} \mid y_i = c_{true} \wedge \hat{y}_i^{(t)} = c_{pred}\}$$

If $|\mathcal{E}_{c_{true} \to c_{pred}}^{(t)}| \ge \theta_{min}$ (where $\theta_{min} = \text{min\_cluster\_size} \ge 1$):
1. A new unique cluster identifier $k_{new} = K^{(t)} + 1$ is allocated.
2. The error subset is excised from its parent cluster and reassigned:
   $$\gamma_i^{(t+1)} = k_{new}, \quad \forall i \in \mathcal{E}_{c_{true} \to c_{pred}}^{(t)}$$
3. The new archetype inherits the true label:
   $$l_{k_{new}} = c_{true}$$
4. Lineage metadata is recorded:
   $$\text{Lineage}(k_{new}) = \left(\text{generation}=t+1, \text{parent}=c_{true}, \text{confused\_with}=c_{pred}, \text{cardinality}=|\mathcal{E}_{c_{true} \to c_{pred}}^{(t)}|\right)$$

In the subsequent generation $t+1$, the centroid of $k_{new}$ is formed:
$$\mu_{k_{new}}^{(t+1)} = \frac{1}{|\mathcal{E}_{c_{true} \to c_{pred}}^{(t)}|} \sum_{i \in \mathcal{E}_{c_{true} \to c_{pred}}^{(t)}} x_i$$

Because these samples were excised from the parent archetype, the parent is **purified** (freed of boundary distortion), while the new archetype provides a dedicated mathematical anchor for that specific confusion boundary.

---

## 3. Algorithm Pseudocode (Formal Algorithm 1)

```
Algorithm 1: Purifying Archetype Classifier (PAC) Training
────────────────────────────────────────────────────────────────────────────────
Input  : Training dataset D = {(x_i, y_i)}_{i=1}^N, x_i in R^D, y_i in {0,...,C-1}
         Max generations G, target accuracy acc_target, min cluster size theta_min
Output : Final archetype set A = {(mu_k, l_k)}_{k=1}^K, Lineage history H

1  // Phase 1: Initialization (Generation 0)
2  for c = 0 to C - 1 do
3      C_c = {i in {1,...,N} | y_i = c}
4      mu_c = (1 / |C_c|) * sum_{i in C_c} x_i
5      l_c = c
6      H[c] = (generation: 0, parent: None, confused_with: None)
7  end
8  A = {(mu_c, l_c)}_{c=0}^{C-1}
9  gamma_i = y_i,  forall i in {1,...,N}
10 PersistentErrors = {1, ..., N}

11 // Phase 2: Purification & Error Bifurcation Loop
12 for t = 0 to G - 1 do
13     // Step 1: Recompute active archetype centroids
14     for each active cluster id k do
15         C_k = {i | gamma_i = k}
16         if |C_k| > 0 then mu_k = (1 / |C_k|) * sum_{i in C_k} x_i
17     end
18     
19     // Step 2: Affinity evaluation
20     for i = 1 to N do
21         k_i* = argmax_{k in A} S(x_i, mu_k)
22         y_hat_i = l_{k_i*}
23     end
24     Correct = {i | y_hat_i = y_i}
25     Accuracy = |Correct| / N
26     PersistentErrors = PersistentErrors \ Correct
27     
28     if Accuracy >= acc_target then break
29     
30     // Step 3: Intra-class purification
31     for each i in Correct do
32         gamma_i = k_i*
33     end
34     
35     // Step 4: Directed confusion bifurcation
36     for each true class c_true in {0,...,C-1} do
37         for each predicted class c_pred in {0,...,C-1} \ {c_true} do
38             E_conf = {i in {1,...,N} \ Correct | y_i = c_true and y_hat_i = c_pred}
39             if |E_conf| >= theta_min then
40                 k_new = allocate_cluster_id()
41                 l_{k_new} = c_true
42                 for each i in E_conf do gamma_i = k_new
43                 H[k_new] = (generation: t+1, parent: c_true, confused_with: c_pred, size: |E_conf|)
44             end
45         end
46     end
47 end
48 return A, H, PersistentErrors
────────────────────────────────────────────────────────────────────────────────
```

---

## 4. Complexity Analysis

### Time Complexity
Let:
- $N$ = Number of training samples
- $D$ = Feature dimensionality
- $G$ = Number of completed generations
- $K^{(t)}$ = Number of active archetypes at generation $t$
- $\bar{K} = \frac{1}{G} \sum_{t=0}^{G-1} K^{(t)}$ = Average number of archetypes across training.

In generation $t$:
1. **Centroid computation:** Partition mean calculation takes $O(N \cdot D)$ time.
2. **Affinity matrix:** Normalizing vectors and computing cosine similarity between $N$ data points and $K^{(t)}$ archetypes takes:
   $$O(N \cdot K^{(t)} \cdot D)$$
3. **Purification & Bifurcation partitioning:** Tensor masking and indexing takes $O(N)$ time.

Summing over $G$ generations, total training time is:
$$\mathcal{T}_{\text{train}} = \sum_{t=0}^{G-1} O(N \cdot K^{(t)} \cdot D) = O(G \cdot N \cdot \bar{K} \cdot D)$$

#### Inference Time
Classifying $M$ test instances against final $K_{final}$ archetypes requires:
$$\mathcal{T}_{\text{inference}} = O(M \cdot K_{final} \cdot D)$$
Compared to $K$-Nearest Neighbors which requires $O(M \cdot N \cdot D)$, PAC provides a speedup factor of:
$$\text{Speedup} = \frac{N}{K_{final}}$$
On MNIST ($N=60,000, K_{final}=1,470$), this represents a theoretical and practical **$40.8\times$** reduction in distance operations. On Fashion-MNIST ($K_{final}=990$), it represents a **$60.6\times$** reduction.

### Space Complexity
PAC stores:
- Input feature matrix: $O(N \cdot D)$
- Archetype matrix: $O(K_{final} \cdot D)$
- Cluster assignment index vector: $O(N)$
Total memory footprint during training is $O(N \cdot D + K_{final} \cdot D)$.  
For deployment (inference), the model size is purely $O(K_{final} \cdot D)$, compressing the stored memory by $\frac{K_{final}}{N}$ compared to KNN.

---

## 5. Technical Rigor & Theoretical Discussion

### Resolution of Technical Inquiries

1. **Intra-class Sub-archetype Dilution:**  
   *Question:* If clean samples migrate into a confusion archetype $\mathcal{E}_{4 \to 9}$, does it dilute the boundary archetype?  
   *Answer:* Yes, and this is by design. If a once-confused archetype $\mu_{4 \to 9}$ attracts clean 4s, its centroid shifts slightly toward the broader manifold of 4s, which expands its decision basin. If subsequent generation errors appear, a child sub-archetype $\mu_{(4 \to 9) \to 7}$ is spawned. This hierarchical refinement prevents catastrophic distortion while stabilizing the boundary.

2. **Bimodal Classes Without Inter-Class Confusion:**  
   *Question:* What happens if class A has two disjoint spatial clusters, but neither causes errors with any other class?  
   *Answer:* PAC does not bifurcate error-free classes. This is an explicit, deliberate property: **PAC allocates capacity proportional to decision boundary complexity, not internal variance.** For density estimation or reconstruction, this is a limitation; for classification and dataset auditing, this is an asset, as it avoids wasting archetypes on internal variances that have no bearing on separability.

3. **Convergencia y Monotonicidad:**  
   Empirically verified across MNIST, Fashion-MNIST, and Tabular datasets, the archetype count $K(t)$ is strictly non-decreasing ($K^{(t+1)} \ge K^{(t)}$) and training accuracy exhibits monotonic convergence until plateauing at the intrinsic noise ceiling of the dataset. Samples that remain perpetually unclassified form the set of **Persistent Errors** $\mathcal{P} = \bigcap_{t=0}^{G-1} \mathcal{E}^{(t)}$, which cleanly isolate mislabeled ground-truth instances.
