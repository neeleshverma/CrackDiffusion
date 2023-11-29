import yaml

from src.utils import get_dataset
from torch.utils.data import DataLoader

with open('configs/ddpm.yaml') as f:
    config = yaml.safe_load(f)

dataset = prepare_dataset(args['training_path'], args['image_size'], args['num_training_images'], args['model_type'])
image_dataloader = DataLoader(dataset, batch_size=args['image_batch_size'], shuffle=True)

for row, (batch_img, batch_label) in tqdm(enumerate(image_dataloader)):
    print(batch_img.shape, batch_label.shape)
    exit(0)