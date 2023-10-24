import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

def parse_image_attribute_labels_to_array(file_path):
    """Parse the image_attribute_labels.txt file into a NumPy array."""
    # <image_id> <attribute_id> <is_present> <certainty_id> <time> content
    
    n_images, n_attributes = 11788, 312
    
    # Initialize the array with zeros
    attribute_matrix = np.zeros((n_images, n_attributes), dtype=np.int8)
    attribute_matrix_cert = np.zeros((n_images, n_attributes), dtype=np.int8)
    
    # Second pass: Populate the array
    with open(file_path, 'r') as file:
        for line in file.readlines():
            image_id, attribute_id, is_present, certainty, _ = map(float, line.strip().split())
            image_id, attribute_id, is_present, certainty = int(image_id), int(attribute_id), int(is_present), int(certainty)
            attribute_matrix[image_id-1, attribute_id-1] = is_present  # -1 because IDs start from 1
            attribute_matrix_cert[image_id-1, attribute_id-1] = certainty  # -1 because IDs start from 1
            
    return attribute_matrix

class CUB2002011Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images, masks, and attributes.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.root_dir = root_dir
        self.transform = transform

        self.image_paths = []
        self.mask_paths = []

        with open(os.path.join(root_dir, "images.txt"), 'r') as file:
            for line in file.readlines():
                image_id, image_relpath = line.strip().split()
                mask_relpath = image_relpath.replace(".jpg", ".png")
                image_path = os.path.join(root_dir, "images", image_relpath)
                mask_path = os.path.join(root_dir, "segmentations", mask_relpath)
                self.image_paths.append(image_path)
                self.mask_paths.append(mask_path)

        attrfile_path = os.path.join(root_dir, "attributes", "image_attribute_labels.txt")
        self.attribute_matrix = parse_image_attribute_labels_to_array(attrfile_path)
        print(self.attribute_matrix.shape)  # Should print (n_images, n_attributes)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Image
        img_path = os.path.join(self.root_dir, "images", self.image_paths[idx])
        image = Image.open(img_path).convert('RGB')
        
        # Segmentation mask
        mask_path = os.path.join(self.root_dir, "segmentations", self.mask_paths[idx])
        mask = Image.open(mask_path).convert('1')  # grayscale
        
        # Attributes
        attributes = self.attribute_matrix[idx, 2:]  # Assuming the attributes are from column 3 onwards

        # to tensor
        # image = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1)
        # mask = torch.tensor(np.array(mask), dtype=torch.float32).unsqueeze(0)
        attributes = torch.tensor(attributes, dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return image, mask, attributes


if __name__ == "__main__":
    # Usage
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    dataset = CUB2002011Dataset(root_dir="/home/longteng/datasets/cub/CUB_200_2011", transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Iterate through the dataset
    for image, mask, attributes in dataloader:
        import pdb
        pdb.set_trace()
        # Your training code here
        pass
