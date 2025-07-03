import torch
from torchvision import transforms
from face_alignment import align
from backbones import get_model

# load model
model_name="edgeface_s_gamma_05" # or edgeface_xs_gamma_06
model=get_model(model_name)
checkpoint_path=f'checkpoints/{model_name}.pt'
model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
model.eval()  # set model to evaluation mode

transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

path = 'assets\FOTU.jpg'
aligned = align.get_aligned_face(path) # align face
transformed_input = transform(aligned) # preprocessing

print(f"Input shape before model: {transformed_input.shape}")

# Check if the tensor is sparse and convert to dense if needed
if transformed_input.is_sparse:
    transformed_input = transformed_input.to_dense()

# Add a batch dimension if it's missing
if len(transformed_input.shape) == 3:
    transformed_input = transformed_input.unsqueeze(0)  # Add batch dimension

print(f"Input shape after adjustments: {transformed_input.shape}") #verify the shape


# extract embedding
embedding = model(transformed_input)