import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import random

from encoder import Encoder

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

# Dataset path
dataset_path = "../periocular_dataset"

# Get all people
people = []

for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)
    if os.path.isdir(person_path) and len(os.listdir(person_path)) > 0:
        people.append(person)

# Function to get embedding
def get_embedding(img_path):
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model(img, return_features=True)
        emb = torch.nn.functional.normalize(emb, dim=1)

    return emb


# Evaluation
num_tests = 100
same_correct = 0
diff_correct = 0
threshold=0.15
for _ in range(num_tests):

    # SAME person
    person = random.choice(people)
    person_path = os.path.join(dataset_path, person)

    images = os.listdir(person_path)
    if len(images) < 2:
        continue

    img1, img2 = random.sample(images, 2)

    emb1 = get_embedding(os.path.join(person_path, img1))
    emb2 = get_embedding(os.path.join(person_path, img2))

    sim = torch.nn.functional.cosine_similarity(emb1, emb2)

    if sim.item() > threshold:
        same_correct += 1

    # DIFFERENT person
    while True:
        p1, p2 = random.sample(people, 2)

        imgs1 = os.listdir(os.path.join(dataset_path, p1))
        imgs2 = os.listdir(os.path.join(dataset_path, p2))

        if len(imgs1) > 0 and len(imgs2) > 0:
            break

    img1 = random.choice(imgs1)
    img2 = random.choice(imgs2)

    emb1 = get_embedding(os.path.join(dataset_path, p1, img1))
    emb2 = get_embedding(os.path.join(dataset_path, p2, img2))

    sim = torch.nn.functional.cosine_similarity(emb1, emb2)

    if sim.item() < threshold:
        diff_correct += 1
    


# Accuracy
same_acc = same_correct / num_tests
diff_acc = diff_correct / num_tests
overall_acc = (same_correct + diff_correct) / (2 * num_tests)

print("\n===== Evaluation Results =====")
print(f"Same Identity Accuracy: {same_acc:.4f}")
print(f"Different Identity Accuracy: {diff_acc:.4f}")
print(f"Overall Accuracy: {overall_acc:.4f}")
print("=============================")