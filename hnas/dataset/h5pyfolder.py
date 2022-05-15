import os
import sys
from PIL import Image

import paddle
from paddle.io import Dataset
from paddle.utils import try_import

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
                  '.tiff', '.webp')
def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def cv2_loader(path):
    cv2 = try_import('cv2')
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def default_loader(path):
    from paddle.vision import get_image_backend
    if get_image_backend() == 'cv2':
        return cv2_loader(path)
    else:
        return pil_loader(path)


def has_valid_extension(filename, extensions):
    """Checks if a file is a vilid extension.

    Args:
        filename (str): path to a file
        extensions (list[str]|tuple[str]): extensions to consider

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    assert isinstance(extensions,
                      (list, tuple)), ("`extensions` must be list or tuple.")
    extensions = tuple([x.lower() for x in extensions])
    return filename.lower().endswith(extensions)

def make_dataset(dir, class_to_idx, extensions, is_valid_file=None):
    images = []
    dir = os.path.expanduser(dir)

    if extensions is not None:

        def is_valid_file(x):
            return has_valid_extension(x, extensions)

    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images

'''
The Official Implementation of DatasetFolder Paddle Paddle
'''
class DatasetFolder(Dataset):
    """A generic data loader where the samples are arranged in this way:

        root/class_a/1.ext
        root/class_a/2.ext
        root/class_a/3.ext

        root/class_b/123.ext
        root/class_b/456.ext
        root/class_b/789.ext

    Args:
        root (string): Root directory path.
        loader (callable|optional): A function to load a sample given its path.
        extensions (list[str]|tuple[str]|optional): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable|optional): A function/transform that takes in
            a sample and returns a transformed version.
        is_valid_file (callable|optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset

    """

    def __init__(self,
                 root,
                 loader=None,
                 extensions=None,
                 transform=None,
                 is_valid_file=None):
        self.root = root
        self.transform = transform
        if extensions is None:
            extensions = IMG_EXTENSIONS
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, extensions,
                               is_valid_file)
        if len(samples) == 0:
            raise (RuntimeError(
                "Found 0 directories in subfolders of: " + self.root + "\n"
                "Supported extensions are: " + ",".join(extensions)))

        self.loader = default_loader if loader is None else loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.dtype = paddle.get_default_dtype()

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), 
                    and class_to_idx is a dictionary.

        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [
                d for d in os.listdir(dir)
                if os.path.isdir(os.path.join(dir, d))
            ]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target, path

    def __len__(self):
        return len(self.samples)


import h5py



def make_dataset_from_H5data(h5data, class_to_idx):
    images = []
    for target in sorted(class_to_idx.keys()):
        for i,np_img in enumerate(h5data[target][target]):
            path = (target,i)  
            item = (path, class_to_idx[target])
            images.append(item)
    return images

class HDF5DatasetFolder(Dataset):
    def __init__(self,
                 root,
                 loader=None,
                 extensions=None,
                 transform=None,
                 is_valid_file=None):
        self.root = root
        self.transform = transform
        self._h5data = h5py.File(self.root, "r")
        classes, class_to_idx = self._find_classes(self._h5data)
        samples = make_dataset_from_H5data(self._h5data, class_to_idx)
        if len(samples) == 0:
            raise (RuntimeError(
                "Found 0 directories in subfolders of: " + self.root + "\n"
                "Supported extensions are: " + ",".join(extensions)))

        self.loader = default_loader if loader is None else loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[-1] for s in samples]
        
        self._h5data = None
        
    @property
    def h5data(self):
        if self._h5data is None: # lazy loading here!
            self._h5data = h5py.File(self.root, "r")
        return self._h5data


    def _find_classes(self, h5_data):
        classes = list(h5_data.keys())
        classes = sorted(classes)
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.h5data[path[0]][path[0]][path[1]]
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def __len__(self):
        return len(self.samples)
