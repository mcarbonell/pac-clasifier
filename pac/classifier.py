"""
Purifying Archetype Classifier (PAC)
A supervised, error-driven prototype spawning classifier and dataset cartography tool.
"""

import time
import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, List, Union


class PurifyingArchetypeClassifier:
    """
    Purifying Archetype Classifier (PAC)
    
    A non-differentiable, prototype-based classification algorithm that dynamically 
    spawns and purifies class centroids ("archetypes") driven by classification errors.
    
    Parameters
    ----------
    max_iters : int, default=200
        Maximum number of purification/bifurcation generations.
    target_acc : float, default=0.999
        Early stopping training accuracy threshold.
    bifurcation_mode : str, default='confusion'
        Error bifurcation strategy:
        - 'confusion': Semantic bifurcation grouping errors by (true_label -> pred_label).
                       Creates interpretable archetypes capturing specific confusion boundaries.
        - 'confidence': Legacy bifurcation sorting errors by confidence and splitting in halves.
    min_cluster_size : int, default=1
        Minimum number of misclassified samples required to spawn a new archetype.
    distance_metric : str, default='cosine'
        Distance/similarity metric: 'cosine' or 'euclidean'.
    device : torch.device or str, optional
        Computation device ('cpu', 'cuda', etc.). If None, automatically selects available hardware.
    """
    def __init__(
        self,
        max_iters: int = 200,
        target_acc: float = 0.999,
        bifurcation_mode: str = 'confusion',
        min_cluster_size: int = 1,
        distance_metric: str = 'cosine',
        device: Optional[Union[str, torch.device]] = None
    ):
        self.max_iters = max_iters
        self.target_acc = target_acc
        self.bifurcation_mode = bifurcation_mode.lower()
        self.min_cluster_size = min_cluster_size
        self.distance_metric = distance_metric.lower()
        
        if self.bifurcation_mode not in ('confusion', 'confidence'):
            raise ValueError(f"Unknown bifurcation_mode '{self.bifurcation_mode}'. Use 'confusion' or 'confidence'.")
        if self.distance_metric not in ('cosine', 'euclidean'):
            raise ValueError(f"Unknown distance_metric '{self.distance_metric}'. Use 'cosine' or 'euclidean'.")

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        # Model state
        self.arch_tensors: Optional[torch.Tensor] = None
        self.arch_labels: Optional[torch.Tensor] = None
        self.arch_cluster_ids: Optional[torch.Tensor] = None
        self.is_fitted: bool = False
        
        # Lineage and interpretability tracking
        self.cluster_history: Dict[int, Dict[str, Any]] = {}
        self.cluster_confusion_map: Dict[int, Tuple[int, int]] = {}
        
        # Dataset auditing diagnostics
        self.never_correct_mask: Optional[torch.Tensor] = None
        self.persistent_error_indices: Optional[torch.Tensor] = None
        self.last_train_preds: Optional[torch.Tensor] = None
        self.last_train_confidences: Optional[torch.Tensor] = None
        
        # Training metrics & trajectory
        self.generation_stats: List[Dict[str, Any]] = []

    def _compute_similarity(self, x: torch.Tensor, archetypes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes sample-archetype affinity matrix.
        Returns:
            best_score: (N,) tensor of affinity scores
            best_idx: (N,) tensor of nearest archetype indices
        """
        if self.distance_metric == 'cosine':
            norm_x = F.normalize(x, p=2, dim=1)
            norm_arch = F.normalize(archetypes, p=2, dim=1)
            sim_matrix = torch.mm(norm_x, norm_arch.t())  # (N, K)
            best_score, best_idx = torch.max(sim_matrix, dim=1)
            return best_score, best_idx
        elif self.distance_metric == 'euclidean':
            # Negative euclidean distance so max corresponds to closest archetype
            dist_matrix = torch.cdist(x, archetypes, p=2.0)  # (N, K)
            min_dist, best_idx = torch.min(dist_matrix, dim=1)
            return -min_dist, best_idx
        else:
            raise NotImplementedError(f"Metric {self.distance_metric} not implemented.")

    def fit(self, x_train: torch.Tensor, y_train: torch.Tensor, verbose: bool = True):
        """
        Fits the PAC model to training data.
        
        x_train: (N, D) tensor of feature vectors (or (N, ...) which will be flattened).
        y_train: (N,) tensor of integer class labels.
        verbose: bool, if True prints progress each generation.
        """
        t0 = time.time()
        
        # Ensure tensor format & flatten spatial dims if needed
        if not isinstance(x_train, torch.Tensor):
            x_train = torch.tensor(x_train, dtype=torch.float32)
        if not isinstance(y_train, torch.Tensor):
            y_train = torch.tensor(y_train, dtype=torch.long)
            
        if x_train.dim() > 2:
            x_train = x_train.reshape(x_train.size(0), -1)
            
        x_train = x_train.float().to(self.device)
        y_train = y_train.long().to(self.device)
        
        N, D = x_train.shape
        
        # Initialize: 1 cluster per class
        unique_classes = torch.unique(y_train)
        num_classes = len(unique_classes)
        
        image_cluster_assignment = y_train.clone()
        next_cluster_id = int(torch.max(unique_classes).item()) + 1
        cluster_to_label = {d.item(): d.item() for d in unique_classes}
        
        # Base lineage tracking
        self.cluster_history = {}
        self.cluster_confusion_map = {}
        self.generation_stats = []
        for d in unique_classes:
            d_val = d.item()
            self.cluster_history[d_val] = {
                'generation': 0,
                'parent': None,
                'true_label': d_val,
                'confused_with': None,
                'size_at_creation': (y_train == d).sum().item()
            }
            
        self.never_correct_mask = torch.ones(N, dtype=torch.bool, device=self.device)
        
        for iteration in range(self.max_iters):
            active_clusters = torch.unique(image_cluster_assignment)
            
            # Step 1: Recompute centroids (archetypes) as pure means of assigned samples
            arch_tensors_list = []
            arch_labels_list = []
            arch_cluster_ids_list = []
            
            for cid in active_clusters:
                cid_val = cid.item()
                mask = (image_cluster_assignment == cid)
                if mask.sum() > 0:
                    arch_tensors_list.append(x_train[mask].mean(dim=0))
                    arch_labels_list.append(cluster_to_label[cid_val])
                    arch_cluster_ids_list.append(cid_val)
                    
            self.arch_tensors = torch.stack(arch_tensors_list)
            self.arch_labels = torch.tensor(arch_labels_list, device=self.device, dtype=torch.long)
            self.arch_cluster_ids = torch.tensor(arch_cluster_ids_list, device=self.device, dtype=torch.long)
            
            # Step 2: Affinity evaluation
            best_scores, best_arch_idx = self._compute_similarity(x_train, self.arch_tensors)
            pred_labels = self.arch_labels[best_arch_idx]
            
            correct = (pred_labels == y_train)
            acc = correct.float().mean().item()
            
            # Update persistent error mask
            self.never_correct_mask[correct] = False
            
            stats = {
                'generation': iteration,
                'num_archetypes': len(active_clusters),
                'train_acc': acc,
            }
            self.generation_stats.append(stats)
            
            if verbose:
                print(f"Gen {iteration:3d} | Active Archetypes: {len(active_clusters):4d} | Train Acc: {acc*100:.2f}%")
                
            # Convergence check
            if acc >= self.target_acc or iteration == self.max_iters - 1:
                break
                
            # Step 3: PURIFY
            # Reassign correctly classified samples to their nearest archetype (intra-class purification)
            image_cluster_assignment[correct] = self.arch_cluster_ids[best_arch_idx[correct]]
            
            # Step 4: BIFURCATE ERRORS
            if self.bifurcation_mode == 'confusion':
                # Semantic bifurcation by confusion pair (true -> predicted)
                for true_d in unique_classes:
                    true_val = true_d.item()
                    base_mask_err = (~correct) & (y_train == true_d)
                    if base_mask_err.sum() == 0:
                        continue
                        
                    for pred_d in unique_classes:
                        pred_val = pred_d.item()
                        if true_val == pred_val:
                            continue
                            
                        mask_err = base_mask_err & (pred_labels == pred_val)
                        misclassified_idx = torch.nonzero(mask_err).squeeze(1)
                        num_errors = len(misclassified_idx)
                        
                        if num_errors >= self.min_cluster_size:
                            image_cluster_assignment[misclassified_idx] = next_cluster_id
                            cluster_to_label[next_cluster_id] = true_val
                            
                            self.cluster_history[next_cluster_id] = {
                                'generation': iteration + 1,
                                'parent': true_val,
                                'true_label': true_val,
                                'confused_with': pred_val,
                                'size_at_creation': num_errors
                            }
                            self.cluster_confusion_map[next_cluster_id] = (true_val, pred_val)
                            next_cluster_id += 1
                            
            elif self.bifurcation_mode == 'confidence':
                # Legacy bifurcation: split errors into two halves by error confidence
                for digit in unique_classes:
                    d_val = digit.item()
                    mask_err = (~correct) & (y_train == digit)
                    misclassified_idx = torch.nonzero(mask_err).squeeze(1)
                    num_errors = len(misclassified_idx)
                    
                    if num_errors > 0:
                        if num_errors == 1:
                            image_cluster_assignment[misclassified_idx] = next_cluster_id
                            cluster_to_label[next_cluster_id] = d_val
                            next_cluster_id += 1
                        else:
                            sims_for_errors = best_scores[misclassified_idx]
                            sorted_args = torch.argsort(sims_for_errors)
                            half = len(sorted_args) // 2
                            
                            h1 = misclassified_idx[sorted_args[:half]]
                            h2 = misclassified_idx[sorted_args[half:]]
                            
                            image_cluster_assignment[h1] = next_cluster_id
                            cluster_to_label[next_cluster_id] = d_val
                            next_cluster_id += 1
                            
                            image_cluster_assignment[h2] = next_cluster_id
                            cluster_to_label[next_cluster_id] = d_val
                            next_cluster_id += 1
                            
        # Final evaluation pass on training data
        final_scores, best_idx_final = self._compute_similarity(x_train, self.arch_tensors)
        self.last_train_confidences = final_scores
        self.last_train_preds = self.arch_labels[best_idx_final]
        self.persistent_error_indices = torch.nonzero(self.never_correct_mask).squeeze(1)
        self.is_fitted = True
        
        return self

    def predict(self, x_test: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Classifies new samples using the learned archetypes.
        Returns:
            predicted_labels: (N,) tensor of class predictions
            affinity_scores: (N,) tensor of similarity/affinity scores
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() first.")
            
        if not isinstance(x_test, torch.Tensor):
            x_test = torch.tensor(x_test, dtype=torch.float32)
        if x_test.dim() > 2:
            x_test = x_test.reshape(x_test.size(0), -1)
            
        x_test = x_test.float().to(self.device)
        best_scores, best_arch_idx = self._compute_similarity(x_test, self.arch_tensors)
        return self.arch_labels[best_arch_idx], best_scores

    def get_archetypes(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the learned archetype tensors and their corresponding labels."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        return self.arch_tensors, self.arch_labels

    def get_archetype_info(self, cluster_id: Optional[int] = None) -> Union[Dict[str, Any], Dict[int, Dict[str, Any]]]:
        """
        Returns lineage and confusion information for archetypes.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
            
        if cluster_id is not None:
            return {
                'cluster_id': cluster_id,
                'label': self.cluster_history.get(cluster_id, {}).get('true_label'),
                'history': self.cluster_history.get(cluster_id),
                'confusion': self.cluster_confusion_map.get(cluster_id)
            }
            
        return {
            cid: {
                'label': info['true_label'],
                'generation': info['generation'],
                'confused_with': info.get('confused_with'),
                'size_at_creation': info.get('size_at_creation')
            }
            for cid, info in self.cluster_history.items()
        }

    def print_confusion_archetypes(self):
        """
        Prints an interpretable summary of all archetypes grouped by their confusion patterns.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
            
        print("\n" + "="*60)
        print("PAC CONFUSION ARCHETYPES SUMMARY")
        print("="*60)
        
        by_true: Dict[int, List[Tuple[int, int]]] = {}
        for cid, (true_l, pred_l) in self.cluster_confusion_map.items():
            if true_l not in by_true:
                by_true[true_l] = []
            by_true[true_l].append((cid, pred_l))
            
        base = [(cid, info['true_label']) for cid, info in self.cluster_history.items() if info['generation'] == 0]
        
        print(f"\nBase Archetypes ({len(base)}):")
        for cid, label in base:
            print(f"  Cluster {cid:3d}: Pure class {label}")
            
        print(f"\nConfusion Archetypes ({len(self.cluster_confusion_map)}):")
        for true_l in sorted(by_true.keys()):
            entries = by_true[true_l]
            print(f"\n  True class '{true_l}' has {len(entries)} confusion variant(s):")
            for cid, pred_l in sorted(entries, key=lambda x: x[1]):
                size = self.cluster_history[cid]['size_at_creation']
                print(f"    Cluster {cid:3d}: {true_l} -> {pred_l} (size at creation: {size})")
                
        print("="*60)

    def predict_with_morphing(
        self,
        x_test: torch.Tensor,
        image_shape: Tuple[int, int] = (28, 28),
        top_k: int = 10,
        morph_iters: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Enhanced 2D prediction using archetype spatial morphing (affine alignment).
        Delegates to pac.morphing module for spatial optimization.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() first.")
            
        from pac.morphing import optimize_morph_fit
        
        if not isinstance(x_test, torch.Tensor):
            x_test = torch.tensor(x_test, dtype=torch.float32)
        if x_test.dim() > 2:
            x_test = x_test.reshape(x_test.size(0), -1)
            
        x_test = x_test.float().to(self.device)
        best_scores, _ = self._compute_similarity(x_test, self.arch_tensors)
        
        # Initial candidates using cosine similarity
        norm_test = F.normalize(x_test, p=2, dim=1)
        norm_arch = F.normalize(self.arch_tensors, p=2, dim=1)
        sim_matrix = torch.mm(norm_test, norm_arch.t())
        
        all_preds = []
        all_sims = []
        
        k_val = min(top_k, len(self.arch_tensors))
        for i in range(x_test.shape[0]):
            img = x_test[i]
            sims = sim_matrix[i]
            _, top_indices = torch.topk(sims, k=k_val)
            
            best_overall_sim = -1.0
            best_label = None
            
            for idx in top_indices:
                arch = self.arch_tensors[idx]
                fit_sim = optimize_morph_fit(
                    img, arch, image_shape=image_shape, iterations=morph_iters, device=self.device
                )
                if fit_sim > best_overall_sim:
                    best_overall_sim = fit_sim
                    best_label = self.arch_labels[idx]
                    
            all_preds.append(best_label)
            all_sims.append(best_overall_sim)
            
        return torch.tensor(all_preds, device=self.device), torch.tensor(all_sims, device=self.device)
