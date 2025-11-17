import numpy as np
from math import lgamma

def euclidean_distances_manual(X, Y):
    """
    Calcula la distancia Euclidiana entre cada par de puntos de X e Y
    """
    X_sq = np.sum(X**2, axis=1, keepdims=True)
    Y_sq = np.sum(Y**2, axis=1, keepdims=True).T
    XY = -2 * (X @ Y.T)
    distances_sq = np.maximum(X_sq + XY + Y_sq, 0.0)
    return np.sqrt(distances_sq)

def silhouette_score(X, labels):
    """
    Calcula el Coeficiente de Silueta (Silhouette Score) promedio
    """
    labels = labels.ravel()
    
    n_samples = len(X)
    if n_samples <= 1:
        return 0
    
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    if n_clusters <= 1:
        return 0
        
    dist_matrix = euclidean_distances_manual(X, X)
    
    silhouette_coeffs = []
    
    for i in range(n_samples):
        my_cluster = labels[i]
        intra_cluster_mask = (labels == my_cluster) & (np.arange(n_samples) != i)
        
        if not np.any(intra_cluster_mask):
            a_i = 0.0
        else:
            a_i = np.mean(dist_matrix[i, intra_cluster_mask])

        min_avg_dist_to_other_cluster = np.inf
        
        for k in unique_labels:
            if k == my_cluster:
                continue
            other_cluster_mask = (labels == k)
            avg_dist_to_k = np.mean(dist_matrix[i, other_cluster_mask])
            
            if avg_dist_to_k < min_avg_dist_to_other_cluster:
                min_avg_dist_to_other_cluster = avg_dist_to_k
        
        b_i = min_avg_dist_to_other_cluster

        if b_i == 0 and a_i == 0:
             s_i = 0.0
        else:
             s_i = (b_i - a_i) / max(a_i, b_i)
             
        silhouette_coeffs.append(s_i)
    
    return np.mean(silhouette_coeffs)

def _remap_labels(labels_true, labels_pred):
    """
    Remapea las etiquetas 'true' y 'pred' a enteros contiguos 
    que empiezan en 0, para que 'np.max' sea seguro
    """
    # Encontrar todas las etiquetas únicas
    true_unique, true_inv = np.unique(labels_true, return_inverse=True)
    pred_unique, pred_inv = np.unique(labels_pred, return_inverse=True)
    
    # Las etiquetas remapeadas son 'true_inv' y 'pred_inv'
    n_classes = len(true_unique)
    n_clusters = len(pred_unique)
    
    return true_inv, pred_inv, n_classes, n_clusters


def _contingency_table(labels_true, labels_pred):
    """
    Construye la "tabla de contingencia"
    """
    true_mapped, pred_mapped, n_classes, n_clusters = _remap_labels(labels_true, labels_pred)
    
    contingency = np.zeros((n_classes, n_clusters), dtype=np.int64)
    
    np.add.at(contingency, (true_mapped, pred_mapped), 1)
        
    return contingency

def _entropy(array):
    """Calcula la entropía de un array de conteos"""
    array = array[array > 0]
    total = np.sum(array)
    probs = array / total
    return -np.sum(probs * np.log(probs))

def homogeneity_completeness_v_measure(labels_true, labels_pred):
    """
    Calcula Homogeneidad, Completitud y V-Measure
    """
    labels_true = labels_true.astype(np.int64).ravel()
    labels_pred = labels_pred.astype(np.int64).ravel()
    
    n_samples = len(labels_true)
    
    # Entropía de las clases (H(C))
    entropy_C = _entropy(np.bincount(labels_true))
    
    # Entropía de los clusters (H(K))
    entropy_K = _entropy(np.bincount(labels_pred))
    
    if entropy_C == 0 and entropy_K == 0:
        return 1.0, 1.0, 1.0
    
    contingency = _contingency_table(labels_true, labels_pred)
    
    H_C_K = 0.0
    for k in range(contingency.shape[1]):
        cluster_k = contingency[:, k]
        if np.sum(cluster_k) > 0:
            H_C_K += np.sum(cluster_k) * _entropy(cluster_k)
    H_C_K /= n_samples
    
    H_K_C = 0.0
    for c in range(contingency.shape[0]):
        class_c = contingency[c, :]
        if np.sum(class_c) > 0:
            H_K_C += np.sum(class_c) * _entropy(class_c)
    H_K_C /= n_samples
    
    if entropy_C == 0:
        homogeneity = 1.0 if H_C_K == 0 else 0.0
    else:
        homogeneity = 1.0 - (H_C_K / entropy_C)
    
    if entropy_K == 0:
        completeness = 1.0 if H_K_C == 0 else 0.0
    else:
        completeness = 1.0 - (H_K_C / entropy_K)
        
    if homogeneity == 0 and completeness == 0:
        v_measure = 0.0
    else:
        v_measure = (2.0 * homogeneity * completeness) / (homogeneity + completeness)
        
    return homogeneity, completeness, v_measure

def normalized_mutual_info_score(labels_true, labels_pred):
    """
    Calcula la Información Mutua Normalizada (NMI)
    """
    labels_true = labels_true.astype(np.int64).ravel()
    labels_pred = labels_pred.astype(np.int64).ravel()
    
    contingency = _contingency_table(labels_true, labels_pred)
    
    N = np.sum(contingency)
    A = np.sum(contingency, axis=1)
    B = np.sum(contingency, axis=0)

    MI = 0.0
    for i in range(contingency.shape[0]):
        for j in range(contingency.shape[1]):
            nij = contingency[i, j]
            if nij > 0:
                # log(x/y) = log(x) - log(y) para estabilidad
                MI += nij * (np.log(N) + np.log(nij) - np.log(A[i]) - np.log(B[j]))
    MI /= N
    
    H_C = _entropy(A)
    H_K = _entropy(B)
    
    if (H_C + H_K) == 0:
        return 1.0 if MI == 0 else 0.0
    
    nmi = (2.0 * MI) / (H_C + H_K)
    return nmi

def _combinations(n, k):
    """Calcula 'n choose k' (combinaciones)"""
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    if k > n // 2:
        k = n - k
    
    return np.exp(lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1))

def adjusted_rand_index(labels_true, labels_pred):
    """
    Calcula el Adjusted Rand Index (ARI)
    """
    labels_true = labels_true.astype(np.int64).ravel()
    labels_pred = labels_pred.astype(np.int64).ravel()
    
    contingency = _contingency_table(labels_true, labels_pred)
    
    sum_nij_c2 = np.sum([_combinations(n, 2) for n in contingency.flat])
    sum_A_c2 = np.sum([_combinations(n, 2) for n in np.sum(contingency, axis=1)])
    sum_B_c2 = np.sum([_combinations(n, 2) for n in np.sum(contingency, axis=0)])
    
    N = len(labels_true)
    total_c2 = _combinations(N, 2)
    
    expected_RI = (sum_A_c2 * sum_B_c2) / total_c2
    max_RI = (sum_A_c2 + sum_B_c2) / 2.0
    
    denominator = (max_RI - expected_RI)
    if denominator == 0:
        return 1.0 if (sum_nij_c2 - expected_RI) == 0 else 0.0
        
    ari = (sum_nij_c2 - expected_RI) / denominator
    return ari