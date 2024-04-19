import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.nn.functional as F
from network_utils import *
from skimage.transform import resize
from utils import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from skimage.transform import resize

# Define VAE architecture
class VAE(nn.Module):
    def __init__(self, input_channels=16, latent_dim=50):
        super(VAE, self).__init__()
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.subcube_h = 1
        self.subcube_w = 1
        self.heights = int(80/ self.subcube_h)
        self.widths = int(80/ self.subcube_w)

        # Encoder layers
        self.conv1 = nn.Conv3d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64*self.heights*self.widths, 512)  # Adjusted the input size to match the output size of conv2
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

        # Decoder layers
        self.fc3 = nn.Linear(latent_dim, 512)
        self.fc4 = nn.Linear(512, 64*self.heights*self.widths)
        self.deconv1 = nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose3d(in_channels=32, out_channels=input_channels, kernel_size=3, stride=1, padding=1)

    def encode(self, x):
        x = x.permute(0, 3, 1, 2)  # Move channels dimension to the second position
        x = x.unsqueeze(4)  # Add a singleton dimension for input channels
        batch_size = x.size(0)  # Get the batch size dynamically
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(batch_size, -1)  # Adjust the view operation for dynamic batch size
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def decode(self, z):
        x = F.relu(self.fc3(z))
        x = F.relu(self.fc4(x))
        x = x.view(-1, 64, self.heights, self.widths)  # Reshape to match the input size before convolutional layers
        x = x.unsqueeze(4)
        x = F.relu(self.deconv1(x))
        x = torch.sigmoid(self.deconv2(x))
        x = x.squeeze(4).permute(0, 2, 3, 1)
        return x
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed_x = self.decode(z)
        return reconstructed_x, mu, logvar

# Define data loading and processing
def load_data(txt_file, hsi_cube_dir, start_wave, end_wave, augment=True):
    # Data Loading
    cropped_hsi_cubes = crop_hsi_cubes(txt_file, hsi_cube_dir, start_wave, end_wave)
    
    # Preprocess Data
    preprocessed_data_list = []
    for hsi_data in cropped_hsi_cubes:
        height, width, depth = hsi_data.shape  

        # Normalize each subcube per wavelength
        max_per_wavelength = np.max(hsi_data, axis=(0, 1), keepdims=True)
        hsi_data_normalized = hsi_data.astype(np.float32) / max_per_wavelength

        # Resize the subcube
        subcube = resize(hsi_data_normalized, (80, 80, hsi_data.shape[2]))

        # Augmentation: flipping and rotation
        if augment:
            # Flipping along both axes
            flipped_subcube = np.flip(subcube, axis=0)
            preprocessed_data_list.append(flipped_subcube)

            flipped_subcube = np.flip(subcube, axis=1)
            preprocessed_data_list.append(flipped_subcube)

            # Rotation
            for angle in [90, 180, 270]:
                rotated_subcube = np.rot90(subcube, k=angle // 90)
                preprocessed_data_list.append(rotated_subcube)
        else:
            preprocessed_data_list.append(subcube)

    preprocessed_data_array = np.array(preprocessed_data_list)
    hsi_tensor = torch.tensor(preprocessed_data_array, dtype=torch.float32)
    del cropped_hsi_cubes
    return hsi_tensor

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = VAE().to(device)
vae.load_state_dict(torch.load('vae_model_cube.pth'))
vae.eval()

# Load and preprocess data
txt_file = "/mnt/c/Users/mahmo/Desktop/Github_Dump/QualiCrop/cropped_test/all_bboxes.txt"
hsi_cube_dir = "/mnt/c/Users/mahmo/Desktop/Github_Dump/test"
start_wave = 529.91
end_wave = 550

#Load Anom
#txt_file = "/mnt/c/Users/mahmo/Desktop/Github_Dump/QualiCrop/cropped_images/all_bboxes.txt"
#hsi_cube_dir = "/mnt/c/Users/mahmo/Desktop/Github_Dump/Dataset"
#start_wave = 529.91
#end_wave = 550

hsi_tensor = load_data(txt_file, hsi_cube_dir, start_wave, end_wave).to(device)
print(hsi_tensor.shape)
# Create DataLoader
dataset = TensorDataset(hsi_tensor)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Initialize lists to store latent embeddings and corresponding file names
latent_embeddings = []
file_names = []

# Extract latent embeddings and file names
with torch.no_grad():
    for i, data in enumerate(dataloader):
        inputs = data[0].to(device)
        _, mu, _ = vae(inputs)
        latent_embeddings.append(mu.cpu().numpy())
        file_names.append("sample_" + str(i))

# Save latent embeddings to a text file
latent_embeddings = np.array(latent_embeddings)
latent_embeddings_flat = latent_embeddings.reshape(latent_embeddings.shape[0], -1)

# Save flattened latent embeddings to a text file
np.savetxt('latent_embeddings.txt', latent_embeddings_flat, delimiter=',')


# Save corresponding file names to a text file
with open('file_names.txt', 'w') as f:
    for file_name in file_names:
        f.write(file_name + '\n')