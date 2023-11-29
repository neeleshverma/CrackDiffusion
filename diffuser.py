import yaml
from tqdm import tqdm
from src.utils import get_dataset
from torch.utils.data import DataLoader

from torchvision import transforms
from diffusers import UNet2DModel

with open('experiments/crack500_2/ddpm.yaml') as f:
    config = yaml.safe_load(f)

# print(config['dim'])
dataset = get_dataset(config['training_path'], config['dim'][0], config['num_training_images'], config['model_type'])
image_dataloader = DataLoader(dataset, batch_size=config['image_batch_size'], shuffle=True)


model = UNet2DModel(
    sample_size=config['dim'][0],  # the target image resolution
    in_channels=3,  # the number of input channels, 3 for RGB images
    out_channels=1,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
)

# print(type())
sample_image = dataset[0][0].unsqueeze(0)
# print("Input shape:", sample_image.shape)
# print("Output shape:", model(sample_image, timestep=0).sample.shape)

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
noise = torch.randn(sample_image.shape)