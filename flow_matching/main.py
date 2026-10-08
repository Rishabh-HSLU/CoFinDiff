from dataset.datalayer import AFHQImageDataset, get_dataloader, NoiseTimeSampler

dataset_path = "/home/rishabh/PycharmProjects/CoFinDiff/data/cat"  # Replace with the actual path to your dataset
# Create an instance of the AFHQImageDataset
dataset = AFHQImageDataset(dataset_path)
# Create a DataLoader for the dataset
dataloader = get_dataloader(dataset, batch_size=1, shuffle=True, num_workers=4)
# Create a NoiseTimeSampler
noise_time_sampler = NoiseTimeSampler(num_samples=1, noise_dim=100)
noise, time = noise_time_sampler.sample()
print("Dataset length:", len(dataset))


