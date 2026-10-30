"""ImageFolder-style dataset loader that builds image path and class-index lists from class directories."""

import os
import os.path

import numpy as np
from PIL import Image
from torchvision.datasets.vision import VisionDataset


def has_file_allowed_extension(filename, extensions):
    return filename.lower().endswith(extensions)


def is_image_file(filename):
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def get_img_from_classes(
    d, images, nimg_cat, class_to_idx, target, is_valid_file, extensions
):
    for root, _, fnames in sorted(os.walk(d)):
        fnames = sorted(fnames)
        if nimg_cat is not None:
            fnames = fnames[:nimg_cat]
        for fname in fnames:
            path = os.path.join(root, fname)
            if is_valid_file(path):
                item = (path, class_to_idx[target])
                images.append(item)
    return images


def make_dataset(
    dir, class_to_idx, addr, nimg_cat, extensions=None, is_valid_file=None
):

    images = []
    dir = os.path.expanduser(dir)
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError(
            "Both extensions and is_valid_file cannot be None or not None at the same time"
        )
    if extensions is not None:

        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)

    if addr != None:
        for target in sorted(addr.keys()):
            d = os.path.join(dir, target)

            if nimg_cat is None:
                for fname in sorted(addr[target]):
                    path = os.path.join(d, fname)
                    if is_valid_file(path):
                        item = (path, class_to_idx[target])
                        images.append(item)
            else:
                images = get_img_from_classes(
                    d, images, nimg_cat, class_to_idx, target, is_valid_file, extensions
                )

    else:
        for target in sorted(class_to_idx.keys()):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            images = get_img_from_classes(
                d, images, nimg_cat, class_to_idx, target, is_valid_file, extensions
            )

    return images


class DatasetFolder(VisionDataset):

    def __init__(
        self,
        root,
        loader,
        address=None,
        cl=None,
        nclasses=None,
        seed=None,
        nimg_cl=None,
        extensions=None,
        transform=None,
        target_transform=None,
        is_valid_file=None,
        class_indices=None,
    ):
        super(DatasetFolder, self).__init__(
            root, transform=transform, target_transform=target_transform
        )
        selectors = sum(
            x is not None for x in (address, cl, class_indices, nclasses)
        )
        if selectors > 1:
            raise TypeError(
                "Provide at most one of address / cl / class_indices / nclasses"
            )
        if address is not None:
            classes, class_to_idx, fname_dict = self._find_classes_from_adr(address)
            samples = make_dataset(
                self.root,
                class_to_idx,
                addr=fname_dict,
                nimg_cat=nimg_cl,
                extensions=extensions,
                is_valid_file=is_valid_file,
            )

        elif cl is not None:
            if not isinstance(cl, list):
                raise TypeError("classes must be list")
            classes = sorted(cl)
            class_to_idx = {classes[i]: i for i in range(len(classes))}
            samples = make_dataset(
                self.root,
                class_to_idx,
                addr=address,
                nimg_cat=nimg_cl,
                extensions=extensions,
                is_valid_file=is_valid_file,
            )

        elif class_indices is not None:
            classes, class_to_idx = self._find_classes_from_indices(
                self.root, class_indices
            )
            samples = make_dataset(
                self.root,
                class_to_idx,
                addr=None,
                nimg_cat=nimg_cl,
                extensions=extensions,
                is_valid_file=is_valid_file,
            )

        else:
            classes, class_to_idx = self._find_classes(self.root, nclasses, seed)
            samples = make_dataset(
                self.root,
                class_to_idx,
                addr=address,
                nimg_cat=nimg_cl,
                extensions=extensions,
                is_valid_file=is_valid_file,
            )

        if len(samples) == 0:
            raise (
                RuntimeError(
                    "Found 0 files in subfolders of: " + self.root + "\n"
                    "Supported extensions are: " + ",".join(extensions)
                )
            )

        self.loader = loader
        self.extensions = extensions
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def _find_classes(self, dir, nclasses, seed):
        if nclasses is None and seed is None:
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]

        elif nclasses is not None and seed is not None:
            all_classes = sorted(d.name for d in os.scandir(dir) if d.is_dir())
            Ntot_classes = len(all_classes)
            if nclasses > Ntot_classes:
                raise ValueError("selected number of classes > available classes")
            np.random.seed(seed)
            chosen_index = np.random.choice(Ntot_classes, nclasses, replace=False)
            classes = [all_classes[i] for i in chosen_index]

        else:
            raise ValueError(
                "nclasses and seed must be None or not None at the same time instead nclasses={} and seed={}".format(
                    nclasses, seed
                )
            )

        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def _find_classes_from_indices(self, dir, class_indices):
        """Select classes by their position in the alphabetically-sorted list of
        class directories under `dir`. Duplicate or out-of-range indices raise."""
        all_classes = sorted(d.name for d in os.scandir(dir) if d.is_dir())
        indices = list(class_indices)
        if len(set(indices)) != len(indices):
            raise ValueError("class_indices contains duplicates")
        max_idx = max(indices) if indices else -1
        if max_idx >= len(all_classes) or min(indices) < 0:
            raise ValueError(
                "class_indices out of range for {} available classes".format(
                    len(all_classes)
                )
            )
        classes = sorted(all_classes[i] for i in indices)
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def _find_classes_from_adr(self, filename):

        with open(filename, "r") as f:
            lines = f.read().splitlines()

        fname_dict = {}
        for line in lines:
            cl = line.split("/")[0]
            img = line.split("/")[1]
            if cl in fname_dict.keys():
                fname_dict[cl].append(img)
            else:
                fname_dict[cl] = [img]

        classes = list(fname_dict.keys())
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx, fname_dict

    def __getitem__(self, index):

        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)


IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def accimage_loader(path):
    import accimage

    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFolder(DatasetFolder):
    def __init__(
        self,
        root,
        address=None,
        cl=None,
        nclasses=None,
        seed=None,
        nimg_cat=None,
        transform=None,
        target_transform=None,
        loader=default_loader,
        is_valid_file=None,
        class_indices=None,
    ):
        super(ImageFolder, self).__init__(
            root,
            loader,
            address,
            cl,
            nclasses,
            seed,
            nimg_cat,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
            class_indices=class_indices,
        )
        self.imgs = self.samples
