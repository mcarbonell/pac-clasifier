import torch
import torch.nn.functional as F

class PurifyingArchetypeClassifier:
    """
    Purifying Archetype Classifier (PAC)
    A supervised clustering and classification algorithm that dynamically spawns
    new archetypes (centroids) based on classification errors, using Cosine Similarity.
    """
    def __init__(self, max_iters=200, target_acc=0.999, device=None):
        self.max_iters = max_iters
        self.target_acc = target_acc
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.arch_tensors = None
        self.arch_labels = None
        self.is_fitted = False
        
    def fit(self, x_train, y_train, verbose=True):
        """
        Trains the PAC algorithm.
        x_train: (N, D) tensor of normalized or unnormalized feature vectors.
        y_train: (N,) tensor of integer labels.
        """
        x_train = x_train.to(self.device)
        y_train = y_train.to(self.device)
        
        N, D = x_train.shape
        
        # Initial assignment: 1 cluster per class
        unique_classes = torch.unique(y_train)
        num_classes = len(unique_classes)
        
        image_cluster_assignment = y_train.clone()
        next_cluster_id = int(torch.max(unique_classes).item()) + 1
        cluster_to_label = {d.item(): d.item() for d in unique_classes}
        
        for iteration in range(self.max_iters):
            active_clusters = torch.unique(image_cluster_assignment)
            
            arch_tensors_list = []
            arch_labels_list = []
            arch_cluster_ids = []
            
            for cid in active_clusters:
                cid = cid.item()
                mask = (image_cluster_assignment == cid)
                if mask.sum() > 0:
                    # The archetype is the mean of its assigned images
                    arch_tensors_list.append(x_train[mask].mean(dim=0))
                    arch_labels_list.append(cluster_to_label[cid])
                    arch_cluster_ids.append(cid)
                    
            self.arch_tensors = torch.stack(arch_tensors_list)
            self.arch_labels = torch.tensor(arch_labels_list, device=self.device)
            arch_cluster_ids = torch.tensor(arch_cluster_ids, device=self.device)
            
            # --- EVALUATE WITH COSINE SIMILARITY ---
            norm_train = F.normalize(x_train, p=2, dim=1)
            norm_arch = F.normalize(self.arch_tensors, p=2, dim=1)
            
            cos_sim = torch.mm(norm_train, norm_arch.t())
            max_sim, best_arch_idx = torch.max(cos_sim, dim=1)
            pred_labels = self.arch_labels[best_arch_idx]
            
            correct = (pred_labels == y_train)
            acc = correct.float().mean().item()
            
            if verbose:
                print(f"Gen {iteration:3d} | Active Archetypes: {len(active_clusters):4d} | Train Acc: {acc*100:.2f}%")
                
            if acc >= self.target_acc or iteration == self.max_iters - 1:
                break
                
            # 1. PURIFY (Reassign correct images to closest valid archetype)
            image_cluster_assignment[correct] = arch_cluster_ids[best_arch_idx[correct]]
            
            # 2. BIFURCATE ERRORS
            for digit in unique_classes:
                digit = digit.item()
                mask_err = (~correct) & (y_train == digit)
                misclassified_indices = torch.nonzero(mask_err).squeeze(1)
                
                num_errors = len(misclassified_indices)
                if num_errors > 0:
                    if num_errors == 1:
                        image_cluster_assignment[misclassified_indices] = next_cluster_id
                        cluster_to_label[next_cluster_id] = digit
                        next_cluster_id += 1
                    else:
                        # Sort by confidence of error
                        sims_for_errors = max_sim[misclassified_indices]
                        sorted_args = torch.argsort(sims_for_errors)
                        half = len(sorted_args) // 2
                        
                        half1_indices = misclassified_indices[sorted_args[:half]]
                        image_cluster_assignment[half1_indices] = next_cluster_id
                        cluster_to_label[next_cluster_id] = digit
                        next_cluster_id += 1
                        
                        half2_indices = misclassified_indices[sorted_args[half:]]
                        image_cluster_assignment[half2_indices] = next_cluster_id
                        cluster_to_label[next_cluster_id] = digit
                        next_cluster_id += 1
                        
        self.is_fitted = True
        
    def predict(self, x_test):
        """
        Classifies new data using the discovered archetypes.
        Returns predictions and the similarity score.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() first.")
            
        x_test = x_test.to(self.device)
        norm_test = F.normalize(x_test, p=2, dim=1)
        norm_arch = F.normalize(self.arch_tensors, p=2, dim=1)
        
        cos_sim_test = torch.mm(norm_test, norm_arch.t())
        max_sim, best_arch_idx = torch.max(cos_sim_test, dim=1)
        
        return self.arch_labels[best_arch_idx], max_sim
        
    def get_archetypes(self):
        """Returns the learned archetype tensors and their corresponding labels."""
        return self.arch_tensors, self.arch_labels
