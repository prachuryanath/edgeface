import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

# Load model using torch.hub
model = torch.hub.load('otroshi/edgeface', 'edgeface_xs_q', 
                      source='github', pretrained=True)
model.eval()

# Define preprocessing transforms
transform = transforms.Compose([
    transforms.Resize((112, 112)),  # EdgeFace typically uses 112x112
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

def get_embedding(image_path, model, transform):
    """Extract embedding from face image"""
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Extract embedding
    with torch.no_grad():
        embedding = model(input_tensor)
        # Normalize embedding (important for cosine similarity)
        embedding = F.normalize(embedding, p=2, dim=1)
    
    return embedding

def calculate_similarity(embedding1, embedding2, method='cosine'):
    """Calculate similarity between two embeddings"""
    if method == 'cosine':
        # Cosine similarity (recommended)
        similarity = F.cosine_similarity(embedding1, embedding2)
        return similarity.item()
    
    elif method == 'euclidean':
        # Euclidean distance (convert to similarity)
        distance = torch.dist(embedding1, embedding2, p=2)
        similarity = 1 / (1 + distance.item())  # Convert distance to similarity
        return similarity
    
# Example usage
image1_path = 'assets\jahi.jpg'
image2_path = 'assets\emon.jpg'

# Extract embeddings
embedding1 = get_embedding(image1_path, model, transform)
embedding2 = get_embedding(image2_path, model, transform)

# Calculate similarity
cosine_sim = calculate_similarity(embedding1, embedding2, 'cosine')
euclidean_sim = calculate_similarity(embedding1, embedding2, 'euclidean')

print(f"Cosine similarity: {cosine_sim:.4f}")
print(f"Euclidean similarity: {euclidean_sim:.4f}")

# Determine if faces match (you'll need to tune these thresholds)
cosine_threshold = 0.4  # Typical range: 0.3-0.6
euclidean_threshold = 0.5

is_same_person_cosine = cosine_sim > cosine_threshold
is_same_person_euclidean = euclidean_sim > euclidean_threshold

print(f"Same person (cosine): {is_same_person_cosine}")
print(f"Same person (euclidean): {is_same_person_euclidean}")
