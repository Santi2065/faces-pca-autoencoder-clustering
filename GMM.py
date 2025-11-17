import numpy as np

class GMM:
    """
    Implementación de GMM con algoritmo EM y Covarianza Diagonal.
    """
    def __init__(self, n_clusters, max_iter=100, tol=1e-4, random_state=42, reg_covar=1e-6):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.rng = np.random.default_rng(self.random_state)
        self.reg_covar = reg_covar
        
        self.weights_ = None
        self.means_ = None
        self.covariances_ = None # (Sigma) - forma (K, D)
        self.log_likelihood_ = -np.inf

    def _initialize_parameters(self, X):
        """Inicializa los parámetros (pi, mu, sigma)."""
        n_samples, n_features = X.shape
        
        self.weights_ = np.full(self.n_clusters, 1.0 / self.n_clusters)
        
        try:
            from kMeans import KMeans
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
            kmeans.fit(X)
            self.means_ = kmeans.centroids
        except Exception:
            indices = self.rng.choice(n_samples, self.n_clusters, replace=False)
            self.means_ = X[indices]
        
        data_var = np.var(X, axis=0) + self.reg_covar
        self.covariances_ = np.array([data_var for _ in range(self.n_clusters)])

    def _multivariate_pdf(self, X, mean, cov_diag):
        """
        Calcula la PDF de una Gaussiana con covarianza DIAGONAL.
        """
        n_features = X.shape[1]
        cov_stable = np.maximum(cov_diag, 1e-10)
        
        log_det = np.sum(np.log(cov_stable))
        log_norm_const = -0.5 * (n_features * np.log(2 * np.pi) + log_det)
        
        diff = X - mean
        exponent_term = (diff**2) / cov_stable
        log_exponent = -0.5 * np.sum(exponent_term, axis=1)
        
        log_prob = log_norm_const + log_exponent
        return np.exp(log_prob)

    def _e_step(self, X):
        """Paso E: Calcula las responsabilidades (soft assignments)."""
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_clusters))
        
        for k in range(self.n_clusters):
            pdf = self._multivariate_pdf(X, self.means_[k], self.covariances_[k])
            responsibilities[:, k] = self.weights_[k] * pdf
        
        sum_responsibilities = np.sum(responsibilities, axis=1, keepdims=True)
        
        zero_prob_mask = (sum_responsibilities == 0).flatten()
        if np.any(zero_prob_mask):
            responsibilities[zero_prob_mask] = 1.0 / self.n_clusters
            sum_responsibilities[zero_prob_mask] = 1.0
            
        responsibilities /= sum_responsibilities
        log_likelihood = np.sum(np.log(np.sum(responsibilities * self.weights_, axis=1)))
        
        return responsibilities, log_likelihood

    def _m_step(self, X, responsibilities):
        """Paso M: Actualiza los parámetros (pi, mu, sigma)."""
        n_samples, n_features = X.shape
        
        Nk = np.sum(responsibilities, axis=0)
        Nk = np.maximum(Nk, 1e-10)
        
        self.weights_ = Nk / n_samples
        self.means_ = (responsibilities.T @ X) / Nk[:, np.newaxis]
        
        for k in range(self.n_clusters):
            diff = X - self.means_[k]
            weighted_sq_diff = responsibilities[:, k, np.newaxis] * (diff**2)
            cov_k_diag = np.sum(weighted_sq_diff, axis=0) / Nk[k]
            self.covariances_[k] = cov_k_diag + self.reg_covar

    def fit(self, X):
        """Entrena el modelo GMM con los datos X usando el algoritmo EM."""
        self._initialize_parameters(X)
        prev_log_likelihood = -np.inf
        
        for i in range(self.max_iter):
            responsibilities, log_likelihood = self._e_step(X)
            self._m_step(X, responsibilities)
            self.log_likelihood_ = log_likelihood
            
            if abs(log_likelihood - prev_log_likelihood) < self.tol:
                break
            
            prev_log_likelihood = log_likelihood
                
    def predict(self, X):
        """Predice la asignación de cluster (hard assignment) para X."""
        if self.means_ is None:
            raise ValueError("El modelo no ha sido entrenado. Llama a .fit() primero.")
        
        responsibilities, _ = self._e_step(X) 
        return np.argmax(responsibilities, axis=1)

    def predict_proba(self, X):
        """Predice las responsabilidades (soft assignment) para X."""
        if self.means_ is None:
            raise ValueError("El modelo no ha sido entrenado. Llama a .fit() primero.")
            
        responsibilities, _ = self._e_step(X)
        return responsibilities