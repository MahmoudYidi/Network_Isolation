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

def find_hsi_cube_file(image_name, hsi_cube_dir):
    image_dir = os.path.join(hsi_cube_dir, image_name, 'capture')
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith(".hdr"):
                #print(os.path.join(root, file))
                return os.path.join(root, file)
    return None 

def crop_hsi_cubes(txt_file, hsi_cube_dir,start_wave, end_wave):
    # Read the bounding box coordinates from the text file
    with open(txt_file, "r") as file:
        bounding_boxes = {}
        for line in file:
            parts = line.strip().split(": ")
            image_name = parts[0].split(".")[0]  # Remove the file extension
            bbox_coords = [float(coord) for coord in parts[1].split()]
            bounding_boxes.setdefault(image_name, []).append(bbox_coords)
    
    # Initialize a list to store cropped HSI cubes
    cropped_hsi_cubes = []
    
    # Iterate over each image name and its corresponding bounding boxes
    for image_name, bboxes in bounding_boxes.items():
        # Find the corresponding HSI cube file
        hsi_cube_file = find_hsi_cube_file(image_name, hsi_cube_dir)
        if hsi_cube_file is None:
            print(f"No HSI cube file found for image: {image_name}")
            continue
        
        # Load the HSI cube
        hsi_data_raw, _ = load_envi_hsi_by_wavelength(hsi_cube_file, start_wave, end_wave)
        #print(hsi_data_raw.shape)
        # Crop each bounding box from the HSI cube
        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            roi = hsi_data_raw[y1:y2, x1:x2, :]
            cropped_hsi_cubes.append(roi)
    del hsi_data_raw
    return cropped_hsi_cubes

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
def loss_function(recon_x, x, mu, logvar, beta=0.8):
    #BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    #BCE = F.mse_loss(recon_x, x, reduction='mean') #Although name is the same, this is MSE!!
    BCE =  F.l1_loss(recon_x, x, reduction='mean') #Same but MAE
    #BCE = F.smooth_l1_loss(recon_x, x, reduction='none') #Huber
    #BCE =1000*F.smooth_l1_loss(recon_x, x, reduction='mean')#smooth L1
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + beta * KLD, BCE, beta * KLD

# Define data loading and processing
def load_data(txt_file, hsi_cube_dir, start_wave, end_wave):
    # Data Loading
    cropped_hsi_cubes = crop_hsi_cubes(txt_file, hsi_cube_dir,start_wave,end_wave)
    
    # Preprocess Data
    preprocessed_data_list = []
    for hsi_data in cropped_hsi_cubes:
        height, width, depth = hsi_data.shape  

        # Normalize each subcube per wavelength
        max_per_wavelength = np.max(hsi_data, axis=(0, 1), keepdims=True)
        hsi_data = hsi_data.astype(np.float32) / max_per_wavelength
        
        subcube = resize(hsi_data, (80, 80, hsi_data.shape[2]))


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

txt_file = "/mnt/c/Users/mahmo/Desktop/Github_Dump/QualiCrop/cropped_test/all_bboxes.txt"
hsi_cube_dir = "/mnt/c/Users/mahmo/Desktop/Github_Dump/test"
start_wave = 529.91
end_wave = 550

# Load and preprocess data
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
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
del dataset #Not sure

def reparam(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std
# Define a function to generate reconstructions

def generate_reconstructions(vae, data):
    with torch.no_grad():
        #recon_batch, _, _ = model(data)
        recon_batch, mu, logvar = vae(data)
        print(reparam(mu,logvar))
        loss, BCE, KLD = loss_function(recon_batch, data, mu, logvar)
    return recon_batch, data, loss

# Define a function to generate samples from the latent space
def generate_samples(model, num_samples=16):
    with torch.no_grad():
        z = torch.randn(num_samples, model.latent_dim).to(device)
        samples = model.decode(z)
    return samples

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = VAE().to(device)
vae.load_state_dict(torch.load('vae_model_cube.pth'))
vae.eval()

# Get a batch of data for visualization
data_iter = iter(dataloader)
data = next(data_iter)[0]
print("data shape: ", data.shape)


# Generate reconstructions
reconstructions, original, loss = generate_reconstructions(vae, data)
print("reconstrcution loss: ", loss.item())
#print("0.13179919123649597")


# Define the figure and subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

wave =10
# Select the wavelength channels
prediction = reconstructions[0][:, :, wave].cpu()
main = original[0][:, :, wave].cpu()

# Get the reflectance values for the main image
H, W, D = original[0].shape
bands = range(1, 17)

# Reshape the main image to a 2D array (H*W) x D
main_reshaped = original[0].cpu().numpy().reshape(-1, D)
pred_reshaped = reconstructions[0].cpu().numpy().reshape(-1, D)

# Compute the mean reflectance per wavelength
mean_reflectance_ori = np.mean(main_reshaped, axis=0)
mean_reflectance_pred = np.mean(pred_reshaped, axis=0)


# Display the first wavelength image
axes[0].imshow(main, cmap='gray')
axes[0].set_title(f"Ground Truth {wave}")
axes[0].set_axis_off()

# Display the second wavelength image
axes[1].imshow(prediction, cmap='gray')
axes[1].set_title(f"Prediction {wave}")
axes[1].set_axis_off()

# Subtract the two images
subtracted_image = prediction - main
subtracted_image =subtracted_image.cpu()


########################################################################################
image_tensor = torch.tensor(subtracted_image, dtype=torch.float32)

# Determine current range
min_val = torch.min(image_tensor)
max_val = torch.max(image_tensor)

# Scale pixel values
norm_image = (image_tensor - min_val) / (max_val - min_val)
subtracted_image= norm_image #Untidy, I know. Just for testing!




# Display the subtracted image
axes[2].imshow(subtracted_image, cmap='gray')
axes[2].set_title("Subtracted Image")
axes[2].set_axis_on()

# Add colorbars
cbar0 = fig.colorbar(axes[0].imshow(main, cmap='gray'), ax=axes[0])
cbar0.set_label('Intensity')

cbar1 = fig.colorbar(axes[1].imshow(prediction, cmap='gray'), ax=axes[1])
cbar1.set_label('Intensity')

cbar2 = fig.colorbar(axes[2].imshow(subtracted_image, cmap='gray'), ax=axes[2])
cbar2.set_label('Intensity Difference')

plt.tight_layout()
plt.show()

A = [-1.8281, -0.0247,  1.2958,  2.1563, -0.4141,  0.8625,  0.8281,  0.3911,
         -2.3022,  0.6075,  0.0391,  1.0571, -0.2072,  0.2157, -1.0049, -1.0555,
          1.1459, -0.0921,  0.7992,  1.4578, -1.7225, -0.4340,  0.9937, -1.5618,
         -0.0038, -1.1866,  0.0262,  1.5763,  0.8909,  0.1436,  0.0263,  1.4977,
         -0.3576, -0.2282, -0.6697, -0.5310,  0.1403,  1.0974,  0.1128, -1.2481,
          0.2754, -0.2593,  0.6472, -2.2767,  1.3172,  0.2286, -1.8433, -0.5958,
          0.8405,  0.5745]

B = [ 1.1328, -1.3847,  1.3986, -0.3614, -0.8100,  0.1672, -1.9450, -0.2905,
          0.3422,  0.1561, -1.8117, -0.0509,  0.1240,  1.0804, -0.1693,  0.3566,
          1.4128,  0.1056,  1.0154,  0.3695, -0.5067, -1.5699,  0.4844, -0.1263,
          0.9114,  0.0281, -0.1311,  2.4302,  0.1891, -0.1930,  0.3233, -1.3028,
          1.6694, -1.8784,  1.1214, -0.1952,  1.1824,  1.7244, -0.5238, -0.5646,
         -1.7152,  1.0810, -1.2718,  1.4069,  0.5590, -1.5088,  0.4131,  0.5111,
         -0.1714,  1.0978]


# Calculate Variance
variance = np.var(A)
variance1 = np.var(B)

# Calculate Standard Deviation
std_dev = np.std(A)
std_dev1 = np.std(B)

print(variance,variance1)
print(std_dev,std_dev1)

# Plot mean reflectance versus wavelength
plt.plot(bands, mean_reflectance_ori, '-')
plt.plot(bands, mean_reflectance_pred, '-')
plt.title('Mean Reflectance vs. Wavelength')
plt.xlabel('Wavelength')
plt.ylabel('Mean Reflectance')
plt.grid(True)
plt.ylim(0, 1)
plt.show()
