# CycleGAN: Image-to-Image Translation Without Paired Data

This project implements a CycleGAN model for unpaired image-to-image translation, specifically for converting images between two domains: horses and zebras. The code supports training the model, generating images, and evaluating results.

---

## Features

- **Image-to-Image Translation**: Convert horse images to zebra images and vice versa.
- **Cycle-Consistency Loss**: Ensures the translations preserve the original content.
- **Pre-trained Models**: Load and utilize pre-trained weights for faster inference.

---

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- albumentations
- matplotlib
- tqdm
- Pillow

---

## Project Structure

```plaintext
├── main.py                # Script for image generation and display
├── train.py               # Training script for CycleGAN
├── models/                # Directory to save/load model weights
├── Horse_Zebra_Images/    # Directory for dataset (trainA, trainB, testA, testB)
├── Generator.py           # Generator model definition
├── Discriminator.py       # Discriminator model definition
├── HorseZebraDataset.py   # Dataset loader for horse and zebra images
```

---

## Usage

### 1. Install Dependencies
Install the required libraries:

```bash
pip install torch torchvision albumentations matplotlib tqdm Pillow
```

### 2. Train the Model

Run the `train.py` script to train the CycleGAN model:

```bash
python train.py
```

The script will:
- **Train the generator and discriminator models.**
- **Save model checkpoints in the `models/` directory.**
- **Display intermediate results during training.**

### 3. Generate Images

Use the `main.py` script to generate images:

```bash
python main.py
```

Follow the on-screen prompts to:
- **Convert zebra images to horses.**
- **Convert horse images to zebras.**

---

## Dataset

```plaintext
Horse_Zebra_Images/
├── trainA/   # Training images of horses
├── trainB/   # Training images of zebras
├── testA/    # Testing images of horses
├── testB/    # Testing images of zebras
```

You can download the Horse to Zebra dataset from [Kaggle](https://www.kaggle.com/datasets/balraj98/horse2zebra-dataset?resource=download).

---

## Key Hyperparameters

| Parameter         | Value      | Description                           |
|-------------------|------------|---------------------------------------|
| Learning Rate     | 1e-5       | Learning rate for optimizers          |
| Batch Size        | 1          | Number of samples per batch           |
| Epochs            | 50         | Total number of training iterations   |
| Image Dimensions  | 256x256    | Resize dimensions for input images    |
| Cycle Loss Weight | 10.0       | Weight for cycle-consistency loss     |

---

## Acknowledgments

- Inspired by the [CycleGAN: Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) paper.
- Special thanks to the [Kaggle community](https://www.kaggle.com/) for providing the Horse to Zebra dataset.
- Implemented using PyTorch and various open-source libraries.


