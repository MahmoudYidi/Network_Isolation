import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import GridSearchCV

# Load the latent embeddings data
latent_embeddings = np.loadtxt('latent_embeddings.txt', delimiter=',')
latent_embeddings1 = np.loadtxt('latent_embeddings1.txt', delimiter=',')

# Define the parameter grid for grid search
param_grid = {
    'n_neighbors': [5, 10, 15, 20,30],  # Experiment with different values
    'metric': ['euclidean', 'manhattan', 'chebyshev']  # Experiment with different distance metrics
}

# Create a LOF model
lof = LocalOutlierFactor(novelty=True)

# Perform grid search to find the optimal parameters
grid_search = GridSearchCV(estimator=lof, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(latent_embeddings)

# Print the best parameters found
print("Best Parameters:", grid_search.best_params_)

# Assuming latent_embeddings1 is your anomalous data
# You can then use the trained model to predict anomalies in the anomalous data
anomaly_scores = -grid_search.best_estimator_.score_samples(latent_embeddings)
anomaly_scores1 = -grid_search.best_estimator_.score_samples(latent_embeddings1)

# Plotting the anomaly scores
plt.figure(figsize=(10, 6))

plt.plot(anomaly_scores, marker='o', label='Normal Data')
plt.plot(anomaly_scores1, marker='o', label='Anomalous Data')

plt.title('Anomaly Scores')
plt.xlabel('Sample Index')
plt.ylabel('Anomaly Score')
plt.legend()
plt.show()