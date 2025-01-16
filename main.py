import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from model import Generator  # Ensure your Generator model is properly imported

# Load device
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load models
gen_H = Generator(img_channels=3, num_residuals=3).to(device)
gen_Z = Generator(img_channels=3, num_residuals=3).to(device)

# Load pre-trained weights
CHECKPOINT_GENERATOR_H = "gen_H.pth"  # Replace with your actual checkpoint file path
CHECKPOINT_GENERATOR_Z = "gen_Z.pth"  # Replace with your actual checkpoint file path

def load_model(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.eval()

load_model(gen_H, CHECKPOINT_GENERATOR_H)
load_model(gen_Z, CHECKPOINT_GENERATOR_Z)

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Function to load and preprocess image
def load_image(filename):
    image = Image.open(filename).convert("RGB")
    return transform(image).unsqueeze(0).to(device)

# Function to display image
def display_image(tensor_image, title):
    image = tensor_image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    image = (image * 0.5 + 0.5).clip(0, 1)  # Unnormalize
    plt.imshow(image)
    plt.axis("off")
    plt.title(title)
    plt.show()

# Main function to prompt user and generate output
def main():
    while True:
        print("Options: (1) Zebra to Horse, (2) Horse to Zebra, (q) Quit")
        choice = input("Enter your choice: ").strip().lower()

        if choice == "q":
            print("Exiting...")
            break

        if choice not in ["1", "2"]:
            print("Invalid choice. Please enter 1, 2, or q.")
            continue

        filename = input("Enter the filename of the image: ").strip()

        try:
            input_image = load_image(filename)
        except Exception as e:
            print(f"Error loading image: {e}")
            continue

        if choice == "1":
            with torch.no_grad():
                output_image = gen_H(input_image)
            display_image(output_image, title="Generated Horse from Zebra")
        elif choice == "2":
            with torch.no_grad():
                output_image = gen_Z(input_image)
            display_image(output_image, title="Generated Zebra from Horse")

if __name__ == "__main__":
    main()