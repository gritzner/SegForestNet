import core
import numpy as np
import rust
import datasets
import types
import cv2 as cv
import torch
import itertools
import utils
from .SparseInstanceImage import SparseInstanceImage
from .SegmentationDatasetPatchProviders import *


class SegmentationDataset():
    def __init__(self, config, params):
        print(f"loading '{config.domain}' images ...")
        config.dataset_path = core.user.dataset_paths[config.domain]
        self.base = getattr(datasets, f"DatasetLoader_{config.domain}")(config)
        assert self.base.channels["gt"][1] == 0
        self.name = config.domain
        self.num_classes = self.base.num_classes
        self.lut = self.base.lut
        self.ignore_class = getattr(config, "ignore_class", -100)
        
        if self.base.gsd != getattr(config, "target_gsd", self.base.gsd):
            self.resample(self.base.gsd / config.target_gsd)
            self.base.gsd = config.target_gsd
        self.gsd = self.base.gsd
        
        self.convert_to_numpy()
        
        if not "depth" in self.base.channels:
            index = 0
            for v in self.base.channels.values():
                index = max(v[0]+1, index)
            self.base.channels["depth"] = (index, 0)
            max_h, max_w = 0, 0
            for img_set in self.base.images:
                img = img_set[0]
                max_h = max(img.shape[0], max_h)
                max_w = max(img.shape[1], max_w)
            img = np.zeros((max_h, max_w), dtype=np.float32)
            for img_set in self.base.images:
                img_set.append(img)
        else:
            assert self.base.channels["depth"][1] == 0
                    
        self.merge_channels()
        self.compute_metadata(config)
        if hasattr(config, "ground_truth_mapping"):
            self.map_ground_truth(config.ground_truth_mapping)

        if self.ignore_class < 0:
            self.ignore_class = self.num_classes
            self.num_classes += 1
            self.lut = list(self.lut)
            self.lut.append((0, 0, 0))
            self.lut = tuple(self.lut)
        
        rng = np.random.RandomState(core.random_seeds[config.random_seed])
        self.split_dataset(rng, config.split_weights)
        self.compute_weights_and_normalization_params(core.device)
        self.create_augmentation(rng, config)
        
        self.training = types.SimpleNamespace(
            x = NormalizedAugmentedInputProvider(self, self.base.input_channels, False, config),
            x_no_rad_aug = NormalizedAugmentedInputProvider(self, self.base.input_channels, True, config),
            x_gt = NormalizedAugmentedInputProvider(self, self.base.gt_channel, False, config),
            x_vis = AugmentedInputProvider(self, config),
            y = AugmentedOutputProvider(self, config)
        )
        
        val_patch_info = self.split_images(*self.split[1:3], config.patch_size, .5 if config.augmentation.at_test_time else 0)
        self.validation = types.SimpleNamespace(
            x = NormalizedInputProvider(self, val_patch_info, self.base.input_channels, config.patch_size, config.augmentation.at_test_time),
            x_gt = NormalizedInputProvider(self, val_patch_info, self.base.gt_channel, config.patch_size, config.augmentation.at_test_time),
            x_vis = InputProvider(self, val_patch_info, config.patch_size, config.augmentation.at_test_time),
            y = OutputProvider(self, val_patch_info, config.patch_size, config.augmentation.at_test_time),
            index_map = IndexMapProvider(val_patch_info, config.patch_size, config.augmentation.at_test_time)
        )
        
        test_patch_info = self.split_images(*self.split[2:4], config.patch_size, .5 if config.augmentation.at_test_time else 0)
        self.test = types.SimpleNamespace(
            x = NormalizedInputProvider(self, test_patch_info, self.base.input_channels, config.patch_size, config.augmentation.at_test_time),
            x_gt = NormalizedInputProvider(self, test_patch_info, self.base.gt_channel, config.patch_size, config.augmentation.at_test_time),
            x_vis = InputProvider(self, test_patch_info, config.patch_size, config.augmentation.at_test_time),
            y = OutputProvider(self, test_patch_info, config.patch_size, config.augmentation.at_test_time),
            index_map = IndexMapProvider(test_patch_info, config.patch_size, config.augmentation.at_test_time)
        )
        
        self.remove_low_entropy_samples(rng, config)
            
        for i, (dataset, label) in enumerate(((self.training, "training"), (self.validation, "validation"), (self.test, "test"))):
            num_pixels = sum([np.prod(img.base.shape[:2]) for img in self.base.images[self.split[i]:self.split[i+1]]])
            if num_pixels >= 10**9:
                num_pixels = f"{num_pixels/10**9:.1f}G"
            elif num_pixels >= 10**6:
                num_pixels = f"{num_pixels/10**6:.1f}M"
            elif num_pixels >= 10**3:
                num_pixels = f"{num_pixels/10**3:.1f}k"
            print(f"# of {label} samples (images, pixels): {dataset.x.shape[0]} ({self.split[i+1]-self.split[i]}, {num_pixels})")
        print(f"input shape: {self.training.x.shape[1:]} -> {self.dtype.base.__name__}")
        print(f"# of classes: {self.num_classes} ({self.dtype.gt.__name__})")
                
    def resample(self, resize_factor):
        for img_set in self.base.images:
            for i, img in enumerate(img_set):
                size = round(img.shape[1]*resize_factor), round(img.shape[0]*resize_factor)
                interp = cv.INTER_NEAREST if i == self.base.channels["gt"][0] else cv.INTER_CUBIC
                img_set[i] = cv.resize(img, size, interpolation=interp)
                    
    def convert_to_numpy(self):
        dtypes = [set(), set()]
        
        for i, img_set in enumerate(self.base.images):
            self.base.images[i] = [img if isinstance(img,SparseInstanceImage) else img.copy() for img in img_set]
            for j, img in enumerate(self.base.images[i]):
                k = 1 if j == self.base.channels["gt"][0] else 0
                if img.dtype != np.float32:
                    dtypes[k].add(img.dtype)
                if len(img.shape) == 2 and img.dtype != np.float32 and k == 0:
                    self.base.images[i][j] = np.expand_dims(img, axis=2)
        
        for i, dtypes_subset in enumerate(dtypes):
            for dtype in dtypes_subset:
                assert dtype in (np.uint8, np.uint16, np.int32)
            dtypes[i] = np.int32 if np.dtype(np.int32) in dtypes_subset else (np.uint16 if np.dtype(np.uint16) in dtypes_subset else np.uint8)        
            if i > 0 and len(dtypes_subset) > 1:
                for img_set in self.base.images:
                    img_set[index] = np.asarray(img_set[self.base.channels["gt"][0]], dtype=dtypes[i])
                
        self.dtype = types.SimpleNamespace(base=dtypes[0], gt=dtypes[1])
        
    def merge_channels(self):
        channels = [c for c in self.base.channels if not c in ("gt", "depth")]
        gt_index = self.base.channels["gt"][0]
        depth_index = self.base.channels["depth"][0]
        for i, img_set in enumerate(self.base.images):
            img = np.empty([*img_set[0].shape[:2], len(channels)], dtype=self.dtype.base)
            for j, c in enumerate(channels):
                indices = self.base.channels[c]
                assert img_set[indices[0]].dtype in (np.uint8, np.uint16, np.int32)
                img[:,:,j] = img_set[indices[0]][:,:,indices[1]]
            assert img_set[depth_index].dtype == np.float32
            self.base.images[i] = types.SimpleNamespace(base=img, gt=img_set[gt_index], depth=img_set[depth_index])
            assert isinstance(self.base.images[i].gt, np.ndarray) or isinstance(self.base.images[i].gt, SparseInstanceImage)
            self.base.images[i].depth -= np.min(self.base.images[i].depth)
        self.base.channels = channels
        
    def compute_metadata(self, config):
        self.base.red_index = self.base.channels.index("red")
        self.base.ir_index = self.base.channels.index("ir") if "ir" in self.base.channels else self.base.red_index
        
        if not hasattr(config.channels, "visualization"):
            if "vis_red" in self.base.channels and "vis_green" in self.base.channels and "vis_blue" in self.base.channels:
                config.channels.visualization = ["vis_red", "vis_green", "vis_blue"]
            elif "red" in self.base.channels and "green" in self.base.channels and "blue" in self.base.channels:
                config.channels.visualization = ["red", "green", "blue"]
            elif "ir" in self.base.channels and "red" in self.base.channels  and "green" in self.base.channels:
                config.channels.visualization = ["ir", "red", "green"]
            else:
                config.channels.visualization = [config.channels.input[0]]
        
        self.base.input_channels = self.get_channel_indices(config.channels.input)
        self.base.visualization_channels = self.get_channel_indices(config.channels.visualization)
        self.base.gt_channel = self.get_channel_indices(["gt"])
        
        self.base.depth_range = np.asarray([np.inf, -np.inf], dtype=np.float32)
        for img in self.base.images:
            self.base.depth_range[0] = min(self.base.depth_range[0], np.min(img.depth))
            self.base.depth_range[1] = max(self.base.depth_range[0], np.max(img.depth))            
        
    def get_channel_indices(self, channels):
        indices = np.empty(len(channels), dtype=np.int32)
        for i, c in enumerate(channels):
            if c == "gt":
                indices[i] = -1
            elif c == "depth":
                indices[i] = -2
            elif c == "ndvi":
                indices[i] = -3
            else:
                indices[i] = self.base.channels.index(c)
        return indices
    
    def map_ground_truth(self, mapping):
        if not (hasattr(mapping, "classes") and hasattr(mapping, "lut")):
            return
        
        assert len(mapping.classes) == self.num_classes
        assert np.min(mapping.classes) >= (0 if isinstance(self.base.images[0].gt,np.ndarray) else 1)
        assert np.max(mapping.classes) < len(mapping.lut)

        self.num_classes = self.base.num_classes = len(mapping.lut)
        for img in self.base.images:
            if isinstance(img.gt, np.ndarray):
                gt = img.gt.copy()
                for src, dst in enumerate(mapping.classes):
                    i = (gt==src).nonzero()
                    img.gt[i[0],i[1]] = dst
            else:
                gt = img.gt.instances[:,0].copy()
                for src, dst in enumerate(mapping.classes):
                    i = (gt==src).nonzero()
                    img.gt.instances[i[0],0] = dst
        self.lut = self.base.lut = tuple([tuple(rgb) for rgb in mapping.lut])
        
    def split_dataset(self, rng, weights):
        if getattr(weights, "override", False) or not hasattr(self.base, "image_subsets"):
            self.base.image_subsets = [0] * len(self.base.images)
        
        assert 0 in self.base.image_subsets
        n_training = 0
        for i in self.base.image_subsets:
            assert i in (0, 1, 2) # training, validation, test
            if i == 0:
                n_training += 1
        
        self.split = np.zeros(4, dtype=np.int32)
        # split[0] = begin of training set
        # split[1] = begin of validation set
        # split[2] = begin of test set
        # split[3] = total number of images
            
        if 1 in self.base.image_subsets:
            assert 2 in self.base.image_subsets
            # training, validation, test already defined
            self.reorder_images(np.argsort(self.base.image_subsets))
            for i in range(1, 3):
                self.split[i] = self.base.image_subsets.index(i)
        elif 2 in self.base.image_subsets:
            # only training and test exist, the later might be validation but treated as test here
            indices = np.argsort(self.base.image_subsets)
            indices[:n_training] = indices[:n_training][rng.permutation(n_training)]
            self.reorder_images(indices)
            
            weight = weights.training / (weights.training + weights.validation)
        
            num_pixels = np.asarray([
                np.prod(img.base.shape[:2]) for img in self.base.images[:n_training]
            ], dtype=np.float64)
            num_pixels = np.cumsum(num_pixels)
            num_pixels /= num_pixels[-1]

            self.split[1] = 1 + np.argmin(np.abs(num_pixels-weight))
            self.split[2] = self.base.image_subsets.index(2)
        else:
            # only training exists        
            self.reorder_images(rng.permutation(len(self.base.images)))
        
            weights = np.asarray([
                weights.training, weights.validation, weights.test
            ], dtype=np.float64)
            weights /= np.sum(weights)
        
            num_pixels = np.asarray([
                np.prod(img.base.shape[:2]) for img in self.base.images
            ], dtype=np.float64)
            num_pixels = np.cumsum(num_pixels)
            num_pixels /= num_pixels[-1]

            self.split[1:3] = 1 + np.argmin(np.abs(num_pixels-weights[0]))
        
            weights = weights[1:]
            weights /= np.sum(weights)        
            num_pixels = num_pixels[self.split[1]:] - num_pixels[self.split[1]-1]
            num_pixels /= num_pixels[-1]
        
            self.split[2] += 1 + np.argmin(np.abs(num_pixels-weights[0]))
        
        self.split[3] = len(self.base.images)

    def reorder_images(self, indices):
        self.base.images = [self.base.images[i] for i in indices]
        self.base.image_subsets = [self.base.image_subsets[i] for i in indices]
        
    def compute_weights_and_normalization_params(self, device):
        self.class_counts = np.zeros(self.num_classes, dtype=np.uint64)
        accum_buffer = (
            np.zeros([len(self.base.channels), 2], dtype=np.uint64),
            np.zeros(2, dtype=np.float64)
        )
        num_pixels = 0
        
        for current_accum_buffer, current_class_counts, current_num_pixels in core.thread_pool.map(self.get_pixel_statistics, range(*self.split[:2])):
            accum_buffer = (accum_buffer[0] + current_accum_buffer[0], accum_buffer[1] + current_accum_buffer[1])
            self.class_counts += current_class_counts
            num_pixels += current_num_pixels
        
        weights = np.empty(self.num_classes, dtype=np.float64)
        for c in range(self.num_classes):
            if self.class_counts[c] == num_pixels:
                weights[c] = 1
            elif c == self.ignore_class or self.class_counts[c] == 0:
                weights[c] = 0
            else:
                weights[c] = 1 - self.class_counts[c]/num_pixels
        weights = weights / np.max(weights)
        self.class_weights = torch.from_numpy(weights).float().to(device)
        
        self.normalization_params = np.zeros([len(self.base.channels)+1, 2], dtype=np.float32)
        self.normalization_params[:-1] = accum_buffer[0] / num_pixels
        self.normalization_params[-1] = accum_buffer[1] / num_pixels
        self.normalization_params[:,1] -= self.normalization_params[:,0]**2
        self.normalization_params[:,1] = np.sqrt(self.normalization_params[:,1])
        np.save(f"{core.output_path}/normalization_params.npy", self.normalization_params)
        
    def get_pixel_statistics(self, i):
        class_counts = np.zeros_like(self.class_counts)
        accum_buffer = (
            np.zeros([len(self.base.channels), 2], dtype=np.uint64),
            np.zeros(2, dtype=np.float64)
        )
        
        img = self.base.images[i]
        if isinstance(img.gt, np.ndarray):
            rust.accumulate_pixel_statistics(accum_buffer[0], accum_buffer[1], class_counts, img.base, img.gt, img.depth)
        else:
            rust.accumulate_pixel_statistics_sparse(accum_buffer[0], accum_buffer[1], class_counts, img.base, img.gt.instances, img.gt.masks, img.depth)
        
        return accum_buffer, class_counts, np.prod(img.base.shape[:2])
        
    def create_augmentation(self, rng, config):
        self.augmentation = types.SimpleNamespace(
            image = np.empty(config.training_samples, dtype=np.int32),
            transforms = np.empty([config.training_samples, 2, 3], dtype=np.float64),
            flips = np.zeros([config.training_samples, 2], dtype=np.uint8),
            radiometric = types.SimpleNamespace(
                contrast = np.empty(config.training_samples, dtype=np.float64),
                brightness = np.empty(config.training_samples, dtype=np.float64),
                seed = np.empty(config.training_samples, dtype=np.uint32),
                noise = np.empty(config.training_samples, dtype=np.float64)
            )
        )
        
        for i in range(config.training_samples):
            self.create_augmented_sample(i, rng, config)
            
    def create_augmented_sample(self, i, rng, config):
        self.augmentation.image[i] = int(rng.uniform(*self.split[:2]))
        img = self.base.images[self.augmentation.image[i]].base
            
        # translation ("cropping")
        T = np.eye(3, dtype=np.float64)
        translation = rng.randint(img.shape[1]-config.patch_size[1]+1), rng.randint(img.shape[0]-config.patch_size[0]+1)
        T[:2,2] = translation

        # horizontal shearing
        val = rng.uniform(-config.augmentation.x_shear, config.augmentation.x_shear)
        T2 = np.eye(3, dtype=np.float64)
        T2[0,1] = np.tan(val*np.pi/180)
        T = T2 @ T

        # vertical shearing
        val = rng.uniform(-config.augmentation.y_shear, config.augmentation.y_shear)
        T2[0,1] = 0
        T2[1,0] = np.tan(val*np.pi/180)
        T = T2 @ T

        # rotation and scaling
        angle = rng.uniform(-config.augmentation.rotation, config.augmentation.rotation)
        scaling = 2**rng.uniform(-config.augmentation.scaling, config.augmentation.scaling)
        T2 = cv.getRotationMatrix2D((translation[0]+.5*config.patch_size[1], translation[1]+.5*config.patch_size[0]), angle, scaling)
        self.augmentation.transforms[i] = T2 @ T

        # horizontal flip
        if rng.rand() < config.augmentation.h_flip:
            self.augmentation.flips[i,0] = 1
            
        # vertical flip
        if rng.rand() < config.augmentation.v_flip:
            self.augmentation.flips[i,1] = 1
            
        # radiometric augmentation (noise, brightness, contrast)
        self.augmentation.radiometric.contrast[i] = 1 + rng.uniform(-config.augmentation.contrast, config.augmentation.contrast)
        self.augmentation.radiometric.brightness[i] = rng.uniform(-config.augmentation.brightness, config.augmentation.brightness)
        self.augmentation.radiometric.seed[i] = rng.randint(np.iinfo(np.uint32).max)
        self.augmentation.radiometric.noise[i] = rng.uniform(0, config.augmentation.noise)
    
    def split_images(self, begin, end, patch_size, overlap):
        assert 0 <= overlap < 1
        overlap = 1 / (1 - overlap)
        result = []
        for i in range(begin, end):
            img = self.base.images[i].base
            yx = [img.shape[j]-patch_size[j] for j in range(2)]
            yx = [np.linspace(0,endpoint,int(np.ceil(overlap*endpoint/patch_size[j]))+1) for j,endpoint in enumerate(yx)]
            yx = [np.asarray(np.round(j), dtype=np.int32) for j in yx]
            for y, x in itertools.product(yx[0], yx[1]):
                result.append((i, y, x))
        return np.asarray(result, dtype=np.int32)
    
    def remove_low_entropy_samples(self, rng, full_config):
        config = full_config.min_sample_entropy
        if config.threshold <= 0:
            assert not config.training_histogram
            return
        
        cache_filename = "datasets/isaid_cache.npz"
        ignore_cache = full_config.domain != "isaid" or getattr(config, "ignore_cache", False)
        create_cache = full_config.domain == "isaid" and getattr(config, "create_cache", False)
        
        entropies = []
        if ignore_cache or create_cache:
            for i in range(self.training.y.shape[0]): # do not parallelize, so that the training set stays deterministic!
                entropy = None
                while True:
                    entropy = self.get_entropy(self.training.y[i])
                    if entropy >= config.threshold:
                        break
                    self.create_augmented_sample(i, rng, full_config)
                entropies.append(entropy)
            if create_cache:
                print(f"creating '{cache_filename}' ...")
                cache = self.augmentation.__dict__.copy()
                cache.update(cache["radiometric"].__dict__.copy())
                del cache["radiometric"]
                np.savez_compressed(cache_filename, **cache)
        else:
            print(f"loading '{cache_filename}' ...")
            cache = np.load(cache_filename)
            for cached_array in cache.files:
                if cached_array in self.augmentation.__dict__:
                    self.augmentation.__dict__[cached_array][:] = cache[cached_array]
                else:
                    self.augmentation.radiometric.__dict__[cached_array][:] = cache[cached_array]
            cache.close()
            
        if config.training_histogram:
            assert ignore_cache or create_cache
            import matplotlib.pyplot as plt
            utils.cdf_plot({"training sample entropy": entropies})
            plt.tight_layout()
            plt.savefig(f"{core.output_path}/training_sample_entropy.pdf")
            
        for dataset, apply in ((self.validation,config.apply_to_validation), (self.test,config.apply_to_test)):
            if not apply:
                continue
            
            keep = []
            for mini_batch in utils.get_mini_batch_iterator(256)(dataset.y):
                for entropy in core.thread_pool.map(lambda i: self.get_entropy(mini_batch[i]), range(mini_batch.shape[0])):
                    keep.append(entropy >= config.threshold)
            keep = np.asarray(keep)
            
            for provider in (dataset.x, dataset.x_gt, dataset.x_vis, dataset.y):
                provider.patch_info = provider.patch_info[keep]
                provider.shape = (provider.patch_info.shape[0], *provider.shape[1:])

    def get_entropy(self, y):
        k = np.asarray([np.count_nonzero(y==c) for c in range(self.num_classes)])
        p = k / np.sum(k)
        p[p<10**-6] = 10**-6
        return -np.sum(p * np.log(p))
    
    def replace_normalization_params(self, new_params):
        assert len(new_params.shape) in (1, 2)
        if len(new_params.shape) == 1:
            new_params = np.expand_dims(new_params, axis=1)
        assert new_params.shape[1] in (1, 2)
        assert self.normalization_params.shape[0] == new_params.shape[0]
        if new_params.shape[1] == 1:
            self.normalization_params[:,1] = new_params[:,0]
        else:
            self.normalization_params[:] = new_params
