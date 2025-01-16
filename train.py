import random, torch, os, copy, sys
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import numpy as np
import Discriminator
import Generator
import HorseZebraDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'using device: {DEVICE}')

HORSE_TRAIN_DIR = 'Horse_Zebra_Images/trainA'
ZEBRA_TRAIN_DIR = 'Horse_Zebra_Images/trainB'
HORSE_VAL_DIR = 'Horse_Zebra_Images/testA'
ZEBRA_VAL_DIR = 'Horse_Zebra_Images/testB'
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0.0 # loss weight for identity loss
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 5
LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_GENERATOR_H = "models/genh.pth.tar"
CHECKPOINT_GENERATOR_Z = "models/genz.pth.tar"
CHECKPOINT_DISCRIMINATOR_H = "models/disch.pth.tar"
CHECKPOINT_DISCRIMINATOR_Z = "models/discz.pth.tar"

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={'image0' : 'image'},
)

def save_checkpoint(model, optimizer, filename):
    print('=> Saving checkpoint')
    checkpoint = {
        'state_dict' : model.state_dict(),
        'optimizer' : optimizer.state_dict(),
    }
    return torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print('=> Loading checkpoint')
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Set deterministic behavior for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False     

def train_fn(disc_H, disc_Z, gen_H, gen_Z, loader, opt_disc, opt_gen, l1, mse):
    H_reals = 0
    H_fakes = 0
    loop = tqdm(loader, leave=True)   

    for idx, (zebra, horse) in enumerate(loop):
        zebra = zebra.to(DEVICE)
        horse = horse.to(DEVICE)

        fake_horse = gen_H(zebra) 
        D_H_real = disc_H(horse)
        D_H_fake = disc_H(fake_horse.detach())
        H_reals += D_H_real.mean().item()
        H_fakes += D_H_fake.mean().item()
        D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real))
        D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake))
        D_H_loss = D_H_fake_loss + D_H_real_loss

        fake_zebra = gen_Z(horse)
        D_Z_real = disc_Z(zebra)
        D_Z_fake = disc_Z(fake_zebra.detach())
        D_Z_real_loss = mse(D_Z_real, torch.ones_like(D_Z_real))
        D_Z_fake_loss = mse(D_Z_fake, torch.zeros_like(D_Z_fake))
        D_Z_loss = D_Z_fake_loss + D_Z_real_loss

        D_loss = (D_H_loss + D_Z_loss) / 2

    opt_disc.zero_grad()
    D_loss.backward()
    opt_disc.step()

    D_H_fake = disc_H(fake_horse)
    D_Z_fake = disc_Z(fake_zebra)
    loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))
    loss_G_Z = mse(D_Z_fake, torch.ones_like(D_Z_fake))

    # cycle losses
    cycle_horse = gen_H(fake_zebra)
    cycle_zebra = gen_Z(fake_horse)
    cycle_horse_loss = l1(horse, cycle_horse)
    cycle_zebra_loss = l1(zebra, cycle_zebra)

    # total loss
    G_loss = loss_G_H + loss_G_Z + cycle_horse_loss + cycle_zebra_loss

    opt_gen.zero_grad()
    G_loss.backward()
    opt_gen.step()

    if idx % 200 == 0:
        save_image(fake_horse * 0.5 + 0.5, f"generated_outputs/horse_{idx}.png")
        save_image(fake_zebra * 0.5 + 0.5, f"generated_outputs/zebra_{idx}.png")

    loop.set_postfix(H_real=H_reals / (idx + 1), H_fake=H_fakes / (idx + 1))

def main():
    disc_H = Discriminator(in_channels=3).to(DEVICE)
    disc_Z = Discriminator(in_channels=3).to(DEVICE)
    gen_H = Generator(img_channels=3, num_residuals=3).to(DEVICE)
    gen_Z = Generator(img_channels=3, num_residuals=3).to(DEVICE)   

    # Use Adam optimizer for both Discriminator and Generator
    opt_disc = optim.Adam(
        disc_H.parameters() + disc_Z.parameters(),
        lr=LEARNING_RATE,
        betas=(0.5, 0.999)
    )    

    opt_gen = optim.Adam(
        list(gen_Z.parameters()) + list(gen_H.parameters()),
        lr=LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if LOAD_MODEL:
        load_checkpoint(
            CHECKPOINT_GENERATOR_H,
            gen_H,
            opt_gen,
            LEARNING_RATE,
        )
        load_checkpoint(
            CHECKPOINT_GENERATOR_Z,
            gen_Z,
            opt_gen,
            LEARNING_RATE,
        )
        load_checkpoint(
            CHECKPOINT_DISCRIMINATOR_H,
            disc_H,
            opt_disc,
            LEARNING_RATE,
        )
        load_checkpoint(
            CHECKPOINT_DISCRIMINATOR_Z,
            disc_Z,
            opt_disc,
            LEARNING_RATE,
        )

    dataset = HorseZebraDataset(
        root_horse=HORSE_TRAIN_DIR,
        root_zebra=ZEBRA_TRAIN_DIR,
        transform=transforms,
    )
    val_dataset = HorseZebraDataset(
        root_horse=HORSE_VAL_DIR,
        root_zebra=ZEBRA_VAL_DIR,
        transform=transforms,
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
    )

    # Training phase
    for epoch in range(NUM_EPOCHS):
        train_fn(disc_H, disc_Z, gen_H, gen_Z, loader, opt_disc, opt_gen, L1, mse,)

    if SAVE_MODEL:
        save_checkpoint(gen_H, opt_gen, filename=CHECKPOINT_GENERATOR_H)
        save_checkpoint(gen_Z, opt_gen, filename=CHECKPOINT_GENERATOR_Z)
        save_checkpoint(disc_H, opt_disc, filename=CHECKPOINT_DISCRIMINATOR_H)
        save_checkpoint(disc_Z, opt_disc, filename=CHECKPOINT_DISCRIMINATOR_Z)    

if __name__ == "__main__":
    main()


