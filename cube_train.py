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
        #self.subcube_h = 10
        #self.subcube_w = 16
        #self.heights = int(800/ self.subcube_h)
        #self.widths = int(1024/ self.subcube_w)
        #
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

#def loss_function(recon_x, x, mu, logvar):
 #   BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
  #  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
   # return BCE + KLD

#Scaled loss now
def loss_function(recon_x, x, mu, logvar, beta=10):
    #BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    #BCE = F.mse_loss(recon_x, x, reduction='mean') #Although name is the same, this is MSE!!
    BCE =  F.l1_loss(recon_x, x, reduction='mean') #Same but MAE
    #BCE = F.smooth_l1_loss(recon_x, x, reduction='none') #Huber
    #BCE =100*F.smooth_l1_loss(recon_x, x, reduction='mean')#smooth L1
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + beta * KLD, BCE, beta * KLD

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

def load_data_pca(txt_file, hsi_cube_dir, start_wave, end_wave,n_components=5):
    # Data Loading
    cropped_hsi_cubes = crop_hsi_cubes(txt_file, hsi_cube_dir,start_wave,end_wave)
    
    # Preprocess Data
    preprocessed_data_list = []
    for hsi_data in cropped_hsi_cubes:
        height, width, depth = hsi_data.shape  

        # Normalize each subcube per wavelength
        max_per_wavelength = np.max(hsi_data, axis=(0, 1), keepdims=True)
        hsi_data = hsi_data.astype(np.float32) / max_per_wavelength
        
        # Flatten the data for PCA
        hsi_data_flattened = hsi_data.reshape(-1, depth)
        
        # Perform PCA on the flattened data
        pca = PCA(n_components=n_components)
        hsi_data_pca = pca.fit_transform(hsi_data_flattened)

        # Normalize PCA output using MinMaxScaler
        scaler = MinMaxScaler()
        hsi_data_pca_normalized = scaler.fit_transform(hsi_data_pca)



        hsi_restored = np.reshape(hsi_data_pca_normalized, (hsi_data.shape[0], hsi_data.shape[1], n_components))

        subcube = resize(hsi_restored, (80, 80, hsi_restored.shape[2]))


        preprocessed_data_list.append(subcube)
    
    preprocessed_data_array = np.array(preprocessed_data_list)
    hsi_tensor = torch.tensor(preprocessed_data_array, dtype=torch.float32)
    del cropped_hsi_cubes
    return hsi_tensor

# Example parameters
#txt_file = "/mnt/c/Users/mahmo/Desktop/Github_Dump/QualiCrop/cropped_images/all_bboxes.txt"
#hsi_cube_dir = "/mnt/c/Users/mahmo/Desktop/Github_Dump/Dataset"
#start_wave = 529.91
#end_wave = 550

txt_file = "/mnt/c/Users/mahmo/Desktop/Github_Dump/QualiCrop/cropped_test/all_bboxes.txt"
hsi_cube_dir = "/mnt/c/Users/mahmo/Desktop/Github_Dump/test"
start_wave = 529.91
end_wave = 550
Use_PCA = False

if Use_PCA:
    hsi_tensor = load_data_pca(txt_file, hsi_cube_dir,start_wave,end_wave,5)
    print(hsi_tensor.shape)
else:
    hsi_tensor = load_data(txt_file, hsi_cube_dir,start_wave,end_wave)
    print(hsi_tensor.shape)
# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create DataLoader
hsi_tensor = hsi_tensor.to(device)
dataset = TensorDataset(hsi_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
del dataset #Not sure
# Instantiate VAE model
vae = VAE().to(device)

# Define optimizer and loss function
optimizer = optim.Adam(vae.parameters(), lr=1e-3)

# Training loop
num_epochs = 100
vae.train()

# Open a file for writing
output_file = open("loss_values.txt", "w")

for epoch in range(num_epochs):
    total_loss = 0
    total_BCE = 0
    total_KLD = 0
    
    for batch_idx, (data,) in enumerate(dataloader):
        optimizer.zero_grad()
        recon_batch, mu, logvar = vae(data)
        loss, BCE, KLD = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_BCE += BCE.item()
        total_KLD += KLD.item()
    
    # Calculate average values
    avg_loss = total_loss / len(dataloader)
    avg_BCE = total_BCE / len(dataloader)
    avg_KLD = total_KLD / len(dataloader)
    
    # Print and write to file
    print('Epoch {}, Average Loss: {:.4f}, Average BCE: {:.4f}, Average KLD: {:.4f}'.format(epoch, avg_loss, avg_BCE, avg_KLD))
    output_file.write('Epoch {}, Average Loss: {:.4f}, Average BCE: {:.4f}, Average KLD: {:.4f}\n'.format(epoch, avg_loss, avg_BCE, avg_KLD))

output_file.close()

torch.save(vae.state_dict(), 'vae_model_cube.pth')