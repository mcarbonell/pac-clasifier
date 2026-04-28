import torch
import torch.nn.functional as F
from torch.optim import Adam

class PurifyingArchetypeClassifierV2:
    """
    Purifying Archetype Classifier V2 (PAC-V2)
    
    Mejora sobre PAC original: la bifurcación de errores se realiza por 
    LABEL DE MISCLASIFICACIÓN en lugar de split arbitrario por confianza.
    
    Esto significa que si un '4' es confundido con '9', todos los '4'→'9'
    forman un nuevo arquetipo. Si otro '4' es confundido con '7', forman
    otro arquetipo distinto. Esto crea clusters semánticamente interpretables:
    cada arquetipo representa una "variante confundida con clase X".
    
    Los nuevos arquetipos se purifican y dividen recursivamente de la misma
    forma en generaciones posteriores.
    """
    def __init__(self, max_iters=200, target_acc=0.999, device=None, min_cluster_size=1):
        self.max_iters = max_iters
        self.target_acc = target_acc
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.min_cluster_size = min_cluster_size
        
        self.arch_tensors = None
        self.arch_labels = None
        self.arch_cluster_ids = None
        self.is_fitted = False
        
        # Tracking de la historia de bifurcación para interpretabilidad
        self.cluster_history = {}
        self.cluster_confusion_map = {}
        
        # Tracking de errores persistentes (para auditoría de datasets)
        self.never_correct_mask = None
        self.persistent_error_indices = None
        self.last_train_preds = None
        self.last_train_confidences = None
        
    def fit(self, x_train, y_train, verbose=True):
        """
        Trains the PAC-V2 algorithm.
        x_train: (N, D) tensor of feature vectors.
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
        
        # Historia: cada cluster nuevo registra de dónde vino
        for d in unique_classes:
            d = d.item()
            self.cluster_history[d] = {
                'generation': 0,
                'parent': None,
                'true_label': d,
                'confused_with': None,  # En generación 0, no hay confusión previa
                'size_at_creation': (y_train == d).sum().item()
            }
        
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
            self.arch_cluster_ids = arch_cluster_ids
            
            # --- EVALUATE WITH COSINE SIMILARITY ---
            norm_train = F.normalize(x_train, p=2, dim=1)
            norm_arch = F.normalize(self.arch_tensors, p=2, dim=1)
            
            cos_sim = torch.mm(norm_train, norm_arch.t())
            max_sim, best_arch_idx = torch.max(cos_sim, dim=1)
            pred_labels = self.arch_labels[best_arch_idx]
            
            correct = (pred_labels == y_train)
            acc = correct.float().mean().item()
            
            # Track persistent errors (never correct in any generation)
            if self.never_correct_mask is None:
                self.never_correct_mask = torch.ones(N, dtype=torch.bool, device=self.device)
            self.never_correct_mask[correct] = False
            
            if verbose:
                print(f"Gen {iteration:3d} | Active Archetypes: {len(active_clusters):4d} | Train Acc: {acc*100:.2f}%")
                
            if acc >= self.target_acc or iteration == self.max_iters - 1:
                break
                
            # 1. PURIFY (Reassign correct images to closest valid archetype)
            image_cluster_assignment[correct] = arch_cluster_ids[best_arch_idx[correct]]
            
            # 2. BIFURCATE ERRORS BY MISCLASSIFICATION LABEL (V2 INNOVATION)
            # Para cada true_label, agrupamos errores por el label al que fueron
            # mal clasificados, y creamos un cluster por cada confusión específica.
            for true_digit in unique_classes:
                true_digit = true_digit.item()
                
                # Solo nos interesan los errores de esta clase verdadera
                base_mask_err = (~correct) & (y_train == true_digit)
                if base_mask_err.sum() == 0:
                    continue
                
                # Para cada posible clase de confusión
                for pred_digit in unique_classes:
                    pred_digit = pred_digit.item()
                    
                    # Saltar si es la misma clase (no sería error)
                    if true_digit == pred_digit:
                        continue
                    
                    # Máscara de errores específicos: true_digit → pred_digit
                    mask_err = base_mask_err & (pred_labels == pred_digit)
                    misclassified_indices = torch.nonzero(mask_err).squeeze(1)
                    num_errors = len(misclassified_indices)
                    
                    # Solo crear cluster si hay suficientes errores de esta confusión
                    if num_errors >= self.min_cluster_size:
                        image_cluster_assignment[misclassified_indices] = next_cluster_id
                        cluster_to_label[next_cluster_id] = true_digit
                        
                        # Registrar historia para interpretabilidad
                        self.cluster_history[next_cluster_id] = {
                            'generation': iteration + 1,
                            'parent': true_digit,  # En V2, el "padre" semántico es la clase verdadera
                            'true_label': true_digit,
                            'confused_with': pred_digit,
                            'size_at_creation': num_errors
                        }
                        self.cluster_confusion_map[next_cluster_id] = (true_digit, pred_digit)
                        
                        next_cluster_id += 1
                        
        # Final evaluation: store predictions and confidences for all training data
        norm_train_final = F.normalize(x_train, p=2, dim=1)
        norm_arch_final = F.normalize(self.arch_tensors, p=2, dim=1)
        cos_sim_final = torch.mm(norm_train_final, norm_arch_final.t())
        self.last_train_confidences, best_idx_final = torch.max(cos_sim_final, dim=1)
        self.last_train_preds = self.arch_labels[best_idx_final]
        self.persistent_error_indices = torch.nonzero(self.never_correct_mask).squeeze(1)
        
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
        
    def _morph_archetype(self, archetype, tx, ty, theta, scale):
        """
        Applies an affine transformation to an archetype.
        archetype: (D,) tensor
        tx, ty: translation offsets
        theta: rotation angle in radians
        scale: zoom factor
        """
        # Reshape to (1, 1, 28, 28) for grid_sample
        img = archetype.view(1, 1, 28, 28)
        
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        
        # Affine matrix:
        # [ scale*cos(theta), -scale*sin(theta), tx ]
        # [ scale*sin(theta),  scale*cos(theta), ty ]
        matrix = torch.stack([
            torch.stack([scale * cos_t, -scale * sin_t, tx]),
            torch.stack([scale * sin_t,  scale * cos_t, ty])
        ]).unsqueeze(0) # (1, 2, 3)
        
        grid = F.affine_grid(matrix, [1, 1, 28, 28], align_corners=False)
        morphed = F.grid_sample(img, grid, align_corners=False)
        
        return morphed.view(-1)

    def _optimize_fit(self, target_img, archetype, iterations=10, lr=0.01):
        """
        Optimizes the transformation parameters to fit the archetype to the target image.
        """
        tx = torch.tensor(0.0, device=self.device, requires_grad=True)
        ty = torch.tensor(0.0, device=self.device, requires_grad=True)
        theta = torch.tensor(0.0, device=self.device, requires_grad=True)
        scale = torch.tensor(1.0, device=self.device, requires_grad=True)
        
        params = [tx, ty, theta, scale]
        optimizer = Adam(params, lr=lr)
        
        best_sim = -1.0
        
        for _ in range(iterations):
            optimizer.zero_grad()
            morphed = self._morph_archetype(archetype, tx, ty, theta, scale)
            
            # We use MSE for the gradient because it's more stable for spatial alignment
            loss = F.mse_loss(morphed, target_img)
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                # Evaluate using cosine similarity to stay consistent with PAC
                sim = F.cosine_similarity(target_img.unsqueeze(0), morphed.unsqueeze(0)).item()
                if sim > best_sim:
                    best_sim = sim
                    
        return best_sim

    def predict_with_morphing(self, x_test, top_k=10, morph_iters=10):
        """
        Enhanced prediction using archetype morphing.
        1. Find top-K closest archetypes.
        2. Optimize transformation for each candidate.
        3. Return the best fitting archetype's label.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() first.")
            
        x_test = x_test.to(self.device)
        norm_test = F.normalize(x_test, p=2, dim=1)
        norm_arch = F.normalize(self.arch_tensors, p=2, dim=1)
        
        # Initial similarity to find candidates
        cos_sim_test = torch.mm(norm_test, norm_arch.t()) # (N, num_arch)
        
        all_preds = []
        all_sims = []
        
        # Process images one by one (morphing is per-image)
        for i in range(x_test.shape[0]):
            img = x_test[i]
            
            # Get top-K candidates
            sims = cos_sim_test[i]
            _, top_indices = torch.topk(sims, k=min(top_k, len(self.arch_tensors)))
            
            best_overall_sim = -1.0
            best_label = None
            
            for idx in top_indices:
                arch = self.arch_tensors[idx]
                # Optimize fit
                fit_sim = self._optimize_fit(img, arch, iterations=morph_iters)
                
                if fit_sim > best_overall_sim:
                    best_overall_sim = fit_sim
                    best_label = self.arch_labels[idx]
            
            all_preds.append(best_label)
            all_sims.append(best_overall_sim)
            
        return torch.tensor(all_preds, device=self.device), torch.tensor(all_sims, device=self.device)

    def get_archetypes(self):
        """Returns the learned archetype tensors and their corresponding labels."""
        return self.arch_tensors, self.arch_labels
    
    def get_archetype_info(self, cluster_id=None):
        """
        Returns interpretable information about archetypes.
        If cluster_id is None, returns info for all archetypes.
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
        
        # Return all
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
        Pretty-prints all archetypes grouped by their confusion patterns.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        
        print("\n" + "="*60)
        print("CONFUSION ARCHETYPES SUMMARY")
        print("="*60)
        
        # Group by true label
        by_true = {}
        for cid, (true_l, pred_l) in self.cluster_confusion_map.items():
            if true_l not in by_true:
                by_true[true_l] = []
            by_true[true_l].append((cid, pred_l))
        
        # Base archetypes (generation 0)
        base = []
        for cid, info in self.cluster_history.items():
            if info['generation'] == 0:
                base.append((cid, info['true_label']))
        
        print(f"\nBase Archetypes ({len(base)}):")
        for cid, label in base:
            print(f"  Cluster {cid:3d}: Pure class {label}")
        
        print(f"\nConfusion Archetypes ({len(self.cluster_confusion_map)}):")
        for true_l in sorted(by_true.keys()):
            entries = by_true[true_l]
            print(f"\n  True class '{true_l}' has {len(entries)} confusion variant(s):")
            for cid, pred_l in sorted(entries, key=lambda x: x[1]):
                size = self.cluster_history[cid]['size_at_creation']
                print(f"    Cluster {cid:3d}: {true_l}→{pred_l} (size at creation: {size})")
        
        print("="*60)
