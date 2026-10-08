from torchvision import transforms
from dataset.datalayer import AFHQImageDataset, get_dataloader, NoiseTimeSampler

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
x0 = noise
ts = 100
dt = 1.0 / ts
