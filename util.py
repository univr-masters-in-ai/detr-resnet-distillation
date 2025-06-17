import torch
import matplotlib.pyplot as plt
def unnormalize(image_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return (image_tensor * std + mean).clamp(0, 1)

def show_image_with_boxes(image_tensor, target, id2label=None):
    image = unnormalize(image_tensor).permute(1, 2, 0).cpu().numpy()

    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    boxes = target["boxes"]
    labels = target["class_labels"]
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        plt.gca().add_patch(
            plt.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=2)
        )
        label = id2label[labels[i].item()] if id2label else str(labels[i].item())
        plt.text(x, y - 5, label['name'], color='white', fontsize=12,
                 bbox=dict(facecolor='red', alpha=0.5))
    plt.axis("off")
    plt.show()