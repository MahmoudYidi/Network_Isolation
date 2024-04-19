import matplotlib.pyplot as plt
import numpy as np
import numpy as np
from scipy.stats import skew, kurtosis


latent_embeddings = np.loadtxt('latent_embeddings.txt', delimiter=',')
latent_embeddings1 = np.loadtxt('latent_embeddings1.txt', delimiter=',')

# Extract x and y coordinates for latent_embeddings
x1 = [sample[0] for sample in latent_embeddings]
y1 = [sample[1] for sample in latent_embeddings]

# Extract x and y coordinates for latent_embeddings1
x2 = [sample[0] for sample in latent_embeddings1]
y2 = [sample[1] for sample in latent_embeddings1]

# Plot latent_embeddings
plt.scatter(x1, y1, label='Normal Data')

# Plot latent_embeddings1
plt.scatter(x2, y2, label='Anomalous Data')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Latent Embeddings')
plt.legend()
plt.show()


# Convert to numpy array for easier computation
latent_embeddings_array = np.array(latent_embeddings)

# Calculate the variance along each dimension (axis=0)
variance = np.var(latent_embeddings_array, axis=0)

print("Variance along each dimension:", variance)

latent_embeddings_array1 = np.array(latent_embeddings1)

# Calculate the variance along each dimension (axis=0)
variance1 = np.var(latent_embeddings_array1, axis=0)

# Dimensions for each dataset
dimensions_latent_embeddings = np.arange(1, len(variance) + 1)
dimensions_latent_embeddings1 = np.arange(1, len(variance1) + 1)

# Plotting
plt.plot(dimensions_latent_embeddings, variance, marker='o', label='Latent Embeddings')
plt.plot(dimensions_latent_embeddings1, variance1, marker='o', label='Latent Embeddings1')

# Adding labels and title
plt.xlabel('Dimension')
plt.ylabel('Variance')
plt.title('Variance of Latent Embeddings and Latent Embeddings1')
plt.legend()

# Display the plot
plt.grid(True)
plt.show()

print(np.mean(variance)*1000000)
print(np.mean(variance1)*1000000)

# Set a threshold to classify anomalies
threshold = 0.3  # Adjust this threshold as needed
anomalies = (variance1*1000000> threshold)

# Print or use the anomalies array as needed
print("Anomalies:", anomalies)