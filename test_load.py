import numpy as np
# Load latent embeddings from the text file
with open('latent_embeddings.txt', 'r') as f:
    lines = f.readlines()

# Convert lines to arrays
latent_embeddings = [np.array(line.strip().split(','), dtype=float) for line in lines]

# Convert list of arrays to a numpy array
latent_embeddings = np.array(latent_embeddings)

print(latent_embeddings[0])


# Generate synthetic test data including anomalies
test_normal_data = np.random.normal(0, 1, size=(num_samples, latent_dim))
test_anomaly_data = np.random.normal(3, 1, size=(num_samples, latent_dim))

# Convert test data to PyTorch tensors
test_normal_data_tensor = torch.tensor(test_normal_data, dtype=torch.float32)
test_anomaly_data_tensor = torch.tensor(test_anomaly_data, dtype=torch.float32)

# Predict anomaly scores for test data
test_normal_scores = subnetwork(test_normal_data_tensor).detach().numpy()
test_anomaly_scores = subnetwork(test_anomaly_data_tensor).detach().numpy()

# Define a threshold for anomaly detection (e.g., using a percentile)
threshold = np.percentile(test_normal_scores, 95)

# Classify instances as normal or anomalous based on the threshold
normal_predictions = test_normal_scores <= threshold
anomaly_predictions = test_anomaly_scores > threshold

# Calculate accuracy, precision, recall, etc.
accuracy = np.mean(normal_predictions)
precision = np.sum(normal_predictions) / (np.sum(normal_predictions) + np.sum(anomaly_predictions))
recall = np.sum(normal_predictions) / len(normal_predictions)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)