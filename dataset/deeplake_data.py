import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets import load_dataset


def collate_fn(examples):
    images = []
    labels = []
    for example in examples:
        images.append((example["pixel_values"]))
        labels.append(example["label"])

    pixel_values = torch.stack(images)
    labels = torch.tensor(labels)
    return pixel_values, labels

def tform(examples):
    transform = transforms.Compose([transforms.ToTensor(),])
    examples["pixel_values"] = [transform(image.convert("RGB")) for image in examples["image"]]
    return examples

def get_tiny_imagenet_data_loader(batch_size=128, num_workers=8):
    train_dataset = load_dataset('Maysee/tiny-imagenet', split='train')
    train_dataset = train_dataset.with_transform(tform)
    train_loader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=batch_size, num_workers=num_workers,
                              shuffle=True)
    val_dataset = load_dataset('Maysee/tiny-imagenet', split='valid')
    val_dataset = val_dataset.with_transform(tform)
    val_loader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=batch_size, num_workers=num_workers)
    return train_loader, val_loader

