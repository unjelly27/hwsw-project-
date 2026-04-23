import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from encoder import Encoder
from periocular_dataset import PeriocularDataset

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = Encoder()
model.load_state_dict(torch.load("periocular_encoder.pth"))
model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# Dataset
dataset = PeriocularDataset("../periocular_dataset", transform)

# Use small subset for visualization
dataset.samples = dataset.samples[:500]
dataset.labels = dataset.labels[:500]

loader = DataLoader(dataset, batch_size=32, shuffle=False)

embeddings = []
labels = []

with torch.no_grad():
    for original, masked40, masked70, label in loader:

        original = original.to(device)

        z = model(original)

        embeddings.append(z.cpu())
        labels.append(label)

embeddings = torch.cat(embeddings).numpy()
labels = torch.cat(labels).numpy()

# t-SNE
tsne = TSNE(n_components=2, random_state=42)
reduced = tsne.fit_transform(embeddings)

# Plot
plt.figure(figsize=(8,6))
scatter = plt.scatter(reduced[:,0], reduced[:,1], c=labels, cmap='tab20')

plt.title("t-SNE Visualization of Periocular Embeddings")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")

plt.show()