from sklearn.neighbors import LocalOutlierFactor
import numpy as np
import matplotlib.pyplot as plt

latent_embeddings = np.loadtxt('latent_embeddings.txt', delimiter=',')
latent_embeddings1 = np.loadtxt('latent_embeddings1.txt', delimiter=',')

# Convert to numpy array for easier computation
latent_embeddings_array = np.array(latent_embeddings)
latent_embeddings_array1 = np.array(latent_embeddings1)

# Calculate the variance along each dimension (axis=0)
variance = np.var(latent_embeddings_array, axis=1)
variance1 = np.var(latent_embeddings_array1, axis=1)

# Assuming latent_embeddings is your normal data
# You can train the LOF model on your normal data
lof = LocalOutlierFactor(novelty=True)
lof.fit(latent_embeddings)
#lof.fit(variance)

# Assuming latent_embeddings1 is your anomalous data
# You can then use the trained model to predict anomalies in the anomalous data
anomaly_scores = -lof.score_samples(latent_embeddings1)
anomaly_scores1 = -lof.score_samples(latent_embeddings)

#anomaly_scores = -lof.score_samples(variance1)
#anomaly_scores1 = -lof.score_samples(variance)

# You can then set a threshold to classify anomalies based on the anomaly scores
threshold = 0  # Adjust this threshold as needed
anomalies = (anomaly_scores > threshold)

# Print or use the anomalies array as needed
print("Anomalies:", anomalies)



# Plotting the anomaly scores
# Plotting the anomaly scores
plt.figure(figsize=(10, 6))

plt.plot(anomaly_scores1, marker='o', label='Normal Data')
plt.plot(anomaly_scores, marker='o', label='Anomalous Data')

plt.title('Anomaly Scores')
plt.xlabel('Sample Index')
plt.ylabel('Anomaly Score')
plt.legend()
plt.show()
# Plot anomaly scores
plt.figure(figsize=(10, 6))
plt.hist(anomaly_scores, bins=50, alpha=0.7, color='blue', label='Anomaly Scores')

# Plot threshold line
plt.axvline(x=threshold, linestyle='--', color='red', label='Threshold')

plt.title('Anomaly Scores Distribution')
plt.xlabel('Anomaly Score')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()


# Plot normal data
plt.figure(figsize=(10, 6))
plt.scatter(latent_embeddings[:, 0], latent_embeddings[:, 1], color='blue', label='Normal Data')

# Plot anomalous data
plt.scatter(latent_embeddings1[:, 0], latent_embeddings1[:, 1], color='red', label='Anomalous Data')

# Highlight detected anomalies
plt.scatter(latent_embeddings1[anomalies][:, 0], latent_embeddings1[anomalies][:, 1], color='orange', label='Detected Anomalies')

plt.title('Anomaly Detection')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()
