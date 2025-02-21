#!/usr/bin/env python3
from pathlib import Path
import random
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import random


def load_image(path_image: str) -> Image.Image:
    """Load image from harddrive and return 3-channel PIL image.
    Args:
        path_image (str): image path
    Returns:
        Image.Image: loaded image
    """
    return Image.open(path_image).convert('RGB')


def get_person_image_paths(path_person_set: str) -> dict:
    """Creates mapping from person name to list of images.
    Args:
        path_person_set (str): Path to dataset that contains folder of images.
    Returns:
        Dict[str, List]: Mapping from person name to image paths,
                         For instance {'name': ['/path/image1.jpg', '/path/image2.jpg']}
    """
    path_person_set = Path(path_person_set)
    person_paths = filter(Path.is_dir, path_person_set.glob('*'))
    return {
        path.name: list(path.glob('*.jpg')) for path in person_paths
    }

def get_persons_with_at_least_k_images(person_paths: dict, k: int) -> list:
    """Filter persons and return names of those having at least k images
    Args:
        person_paths (dict): dict of persons, as returned by `get_person_image_paths`
        k (int): number of images to filter for

    Returns:
        list: list of filtered person names
    """
    return [name for name, paths in person_paths.items() if len(paths) >= k]

  
class TripletFaceDataset(Dataset):

    def __init__(self, path) -> None:
        super().__init__()

        self.person_paths = get_person_image_paths(path)
        # self.persons = self.person_paths.keys()
        self.persons = list(self.person_paths.keys())
        print('--->',self.persons)
        # self.persons_positive = get_persons_with_at_least_k_images(self.person_paths, 2)
        self.persons_positive = list(get_persons_with_at_least_k_images(self.person_paths, 2))
        print('----->',self.persons_positive)
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def get_anchor_positive_negative_paths(self, index: int) -> tuple:
        """Randomly sample a triplet of image paths.

        Args:
            index (int): Index of the anchor / positive person.

        Returns:
            tuple[Path]: A triplet of paths (anchor, positive, negative)
        """
        # TODO Please implement this function
        person = list(self.persons_positive)[index] # get the anchor person
        print('---->',person)
        list_person_img = self.person_paths[person][:] # images of the same person
        a = random.choice(list_person_img)
        list_person_img.remove(a) # avoid taking the same image as positive
        p = random.choice(list_person_img) # get a positive example
        # get a negative person
        n_person = random.choice(list(self.persons))
        while n_person == person:
            # n_person = random.choice(list(self.persons))
            n_person = random.choice(self.persons)
        n = random.choice(self.person_paths[n_person]) # get a negative example

        return a, p, n

    def __getitem__(self, index: int):
        """Randomly sample a triplet of image tensors.

        Args:
            index (int): Index of the anchor / positive person.

        Returns:
            tuple[Path]: A triplet of tensors (anchor, positive, negative)
        """
        a, p, n = self.get_anchor_positive_negative_paths(index)
        return (
            self.transform(load_image(a)),
            self.transform(load_image(p)),
            self.transform(load_image(n))
        )

    def __len__(self):
        return len(self.persons_positive)


