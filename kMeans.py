import numpy as np

class KMeans:

    def __init__(self, n_clusters, max_iter=300, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
        self.inertia_ = 0

    def _initialize_centroids(self, X):
        """Inicializa los centroides eligiendo K puntos aleatorios de X"""
        # Creamos un generador de números aleatorios con la semilla
        rng = np.random.default_rng(self.random_state)
        n_samples, _ = X.shape
        
        # Elegimos 'n_clusters' índices únicos
        indices = rng.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[indices]

    def _assign_clusters(self, X):
        """
        Asigna cada punto en X al centroide más cercano.
        Devuelve un array de etiquetas (índices de cluster)
        """
        # (N, 1, D) - (K, D) -> (N, K, D)
        distances = X[:, np.newaxis] - self.centroids
        
        # Distancia Euclidiana al cuadrado
        sq_distances = np.sum(distances**2, axis=2)
        
        # Asignamos el índice (0 a K-1) del centroide más cercano
        return np.argmin(sq_distances, axis=1)

    def _update_centroids(self, X, labels):
        """Recalcula los centroides como la media de los puntos asignados"""
        new_centroids = np.zeros_like(self.centroids)
        for k in range(self.n_clusters):
            # Obtenemos los puntos asignados a este cluster 'k'
            points_in_cluster = X[labels == k]
            
            # Manejamos el caso de un cluster vacío
            if len(points_in_cluster) > 0:
                new_centroids[k] = points_in_cluster.mean(axis=0)
            else:
                # Re-inicializamos el centroide si el cluster queda vacío
                # Elige un punto aleatorio de X
                rng = np.random.default_rng()
                new_centroids[k] = X[rng.choice(X.shape[0])]
                
        return new_centroids

    def _calculate_inertia(self, X, labels):
        """Calcula la Inercia (SSE): suma de distancias al cuadrado"""
        inertia = 0
        for k in range(self.n_clusters):
            points_in_cluster = X[labels == k]
            if len(points_in_cluster) > 0:
                # Suma de distancias al cuadrado de los puntos a su centroide
                inertia += np.sum((points_in_cluster - self.centroids[k])**2)
        return inertia

    def fit(self, X):
        """
        Entrena el modelo K-Means con los datos X.
        """
        self._initialize_centroids(X)
        
        for i in range(self.max_iter):
            # Asignar clusters (E-Step)
            labels = self._assign_clusters(X)
            
            # Actualizar centroides (M-Step)
            new_centroids = self._update_centroids(X, labels)
            
            self.centroids = new_centroids
        
        self.labels_ = self._assign_clusters(X)
        self.inertia_ = self._calculate_inertia(X, self.labels_)

    def predict(self, X):
        """
        Predice el cluster más cercano para cada muestra en X.
        """
        return self._assign_clusters(X)