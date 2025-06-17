import os
import json
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
from transformers import DetrImageProcessor
from functools import partial
from torchvision.transforms.functional import to_tensor

import torchvision.transforms.functional as F

def resize_boxes(boxes, orig_size, new_size):
    if len(boxes) == 0:
        return boxes
    
    orig_h, orig_w = orig_size
    new_h, new_w = new_size

    scale_w = new_w / orig_w
    scale_h = new_h / orig_h


    resized_boxes = boxes.clone()
    resized_boxes[:, 0] *= scale_w
    resized_boxes[:, 1] *= scale_h 
    resized_boxes[:, 2] *= scale_w
    resized_boxes[:, 3] *= scale_h

    return resized_boxes


def simulate_processor(image):
    """
    Mimics what DetrImageProcessor would return,
    but skips resizing, normalization, etc.
    """
    tensor = to_tensor(image)  # Converts to [C, H, W] in [0, 1] range
    return {
        "pixel_values": tensor.unsqueeze(0)  # Add batch dimension: [1, 3, H, W]
    }


def crop_or_pad_top_left(image, target_size=(512, 512)):
    """
    Crop or pad an image starting from the top-left corner to a fixed size.
    Pads with 0 (black).
    """
    w, h = image.size
    tw, th = target_size

    # Crop to target size
    cropped = image.crop((0, 0, min(w, tw), min(h, th)))

    # Pad to target size (if needed)
    padded = ImageOps.pad(cropped, size=target_size, centering=(0, 0))  # Top-left origin
    return padded

def collate_fn(batch, processor):
    pixel_values = [item["pixel_values"].squeeze(0) for item in batch]
    labels = [item["labels"][0] for item in batch]

    encoding = processor.pad(pixel_values, return_tensors="pt")

    return {
        "pixel_values": encoding["pixel_values"],
        "pixel_mask": encoding["pixel_mask"],
        "labels": labels
    }



class DETRDataset(Dataset):
    def __init__(self, processor, image_dir="../subset/train/images", annotation_file=None):
        self.processor = processor
        self.image_dir = image_dir

        if annotation_file is None:
            raise ValueError("You must provide an annotation JSON file.")

        with open(annotation_file, "r") as f:
            data = json.load(f)

        self.images = data["images"]         
        self.annotations = data["annotations"]
        self.categories= data["categories"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_info = self.images[idx]
        annotation_info = self.annotations[idx]

        image_id = image_info["id"]
        file_name = image_info["file_name"]
        image_path = os.path.join(self.image_dir, file_name)

        image = Image.open(image_path).convert("RGB")

        anns = annotation_info.get("annotations", [])
        boxes = [ann["bbox"] for ann in anns]
        labels = [ann["category_id"] for ann in anns]

        target = {
            "image_id": torch.tensor([image_id]),
            "class_labels": torch.tensor(labels, dtype=torch.int64),
            "boxes": torch.tensor(boxes, dtype=torch.float32)
        }
        orig_size = image.size[::-1]
        inputs = self.processor(image, return_tensors="pt")
        new_size = inputs["pixel_values"].shape[-2:] 
        boxes = resize_boxes(target['boxes'], orig_size, new_size)
        target["boxes"] = boxes
        inputs["labels"] = [target]
        return inputs

def get_train_dataset():
    return get_dataset('../subset/train/images', './train.json')

def get_dataset(image_dir, annotation_file):
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", size=None)
    return DETRDataset(processor=processor, image_dir=image_dir, annotation_file=annotation_file)

def get_test_dataset():
    return get_dataset('../subset/test/images', './test.json')

def get_dataloader(dataset,batch_size, shuffle, num_workers):
    collate = partial(collate_fn, processor=dataset.processor)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate
    )

def get_train_dataloader(batch_size, shuffle, num_workers):
    dataset= get_train_dataset()
    return get_dataloader(dataset, batch_size, shuffle, num_workers)

def get_test_dataloader(batch_size, shuffle, num_workers):
    dataset= get_test_dataset()
    return get_dataloader(dataset, batch_size, shuffle, num_workers)
