import json
import random

def balance_dataset(input_json_path, output_json_path, negatives_ratio=0.5):
    """
    Reads a JSON annotation file, keep all positive images (i.e. images with an annotation)
    and a random sample of 50% of the negative images.
    """
    with open(input_json_path, 'r') as f:
        data = json.load(f)

    annotations = data['annotations']
    images = data['images']
    categories = data['categories']

    # Find positives images IDs
    positive_image_ids = set()
    for i in annotations:
        if i.get('annotations'):
            positive_image_ids.add(i['image_id'])

    print(f"Found {len(positive_image_ids)} positive images.")

    # Split positives and negative images
    positive_images = []
    negative_images = []
    for img in images:
        if img['id'] in positive_image_ids:
            positive_images.append(img)
        else:
            negative_images.append(img)

    num_negatives = len(negative_images)
    print(f"Found {num_negatives} negative images.")

    # Random sample half of the negative images
    num_negatives_keep = int(num_negatives * negatives_ratio)
    negative_img_sampled = random.sample(negative_images, num_negatives_keep)
    print(f"Sampling {len(negative_img_sampled)} negative images.")

    # Final img + anno list
    new_train_images = positive_images + negative_img_sampled
    random.shuffle(new_train_images)
    id_img_keep = {img['id'] for img in new_train_images}
    new_anno = [ann for ann in annotations if ann['image_id'] in id_img_keep]

    # New json file
    new_train_ds = {
        'images': new_train_images,
        'annotations': new_anno,
        'categories': categories
    }

    print(f"Saving new train annotation file in: {output_json_path}")
    with open(output_json_path, 'w') as f:
        json.dump(new_train_ds, f, indent=4)

    print(f"New dataset dimension: {len(new_train_images)} total images.")


if __name__ == '__main__':
    input_path = 'subset/train.json'
    output_path = 'subset/train_sampled.json'
    perc_to_keep = 0.5
    balance_dataset(input_path, output_path, perc_to_keep)