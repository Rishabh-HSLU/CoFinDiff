import torch
import torch.nn.functional as F
from torchvision import transforms
from dataset.datalayer import AFHQImageDataset, get_dataloader, NoiseTimeSampler
from flow.flow import ConditionalVectorField, TargetVectorField

dataset_path = "/home/rishabh/PycharmProjects/CoFinDiff/data/cat"  # Replace with the actual path to your dataset
# Create an instance of the AFHQImageDataset
transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
dataset = AFHQImageDataset(dataset_path, transform=transform)

# Create a DataLoader for the dataset
dataloader = get_dataloader(dataset, batch_size=1, shuffle=True, num_workers=4)



# Create a NoiseTimeSampler
image_dim = dataset.__getitem__(0).flatten().shape[0]
noise_time_sampler = NoiseTimeSampler(num_samples=1, noise_dim= image_dim)
noise, time = noise_time_sampler.sample()



# Initialize the base state
num_epochs = 10
ts = 100
dt = 1.0 / ts



# Training loop

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConditionalVectorField(in_dim=image_dim, z_dim=image_dim, time_emb_dim=128, hidden_dim=256).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
for epoch in range(num_epochs):
    for p_data in dataloader:
        batch_size = p_data.size(0)
        p_data = p_data.to(device)


        z = p_data.clone().flatten(start_dim=1)
        noise, t = noise_time_sampler.sample()
        noise = noise.to(device)
        t = t.to(device)

        target_field = TargetVectorField(p_init=noise, p_data=z)
        xt, v_target = target_field.forward(t=t)

        v_target = v_target.detach()
        xt = xt.detach()

        optimizer.zero_grad()
        v_pred = model(xt, z, t)

        loss = F.mse_loss(v_pred, v_target)
        loss.backward()
        optimizer.step()



# Generation

import matplotlib.pyplot as plt

# 1. Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.eval()  # Set model to evaluation mode

# We need the target image 'z' we want to morph into
# Assuming dataloader yields (p_data) and we grab one batch
p_data = next(iter(dataloader))
p_data = p_data.to(device)
z = p_data.clone().flatten(start_dim=1)

batch_size = z.shape[0]
num_steps = 100
dt = 1.0 / num_steps

# 2. Initialize the starting state at t=0 (Pure Noise)
xt = torch.randn_like(z).to(device)

# 3. The Euler Integration Loop
with torch.no_grad():
    for step in range(num_steps):
        # Format the continuous time scalar into a batched tensor
        current_t = step * dt
        t_tensor = torch.full((batch_size, 1), current_t, device=device)

        # Query the network for the velocity
        v_pred = model(xt, z, t_tensor)

        # The Euler Step: position = position + velocity * time
        xt = xt + (v_pred * dt)

# 4. Unflatten and Render the Final Image
# xt is currently (batch_size, flattened_dim). Reshape it back to an image.
channels, height, width = 3, 256, 256
final_image = xt.view(batch_size, channels, height, width)

# Clamp values to valid image range [-1, 1] and shift to [0, 1] for viewing
final_image = torch.clamp(final_image, -1.0, 1.0)
final_image = (final_image + 1.0) / 2.0

# Plot the first image in the batch
plt.imshow(final_image[0].cpu().permute(1, 2, 0))
plt.axis('off')
plt.show()