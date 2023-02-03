import numpy as np
import core
import rust


class AbstractPatchProvider():
    def __getitem__(self, key):
        if isinstance(key, tuple):
            t = type(key[0])
            result = self.extract_patches(key[0])
            result = result.__getitem__((slice(result.shape[0]), *key[1:]))
        else:
            t = type(key)
            result = self.extract_patches(key)
        return result if t==np.ndarray or t==list or t==slice else result[0]
    
    def extract_patches(self, indices):
        if isinstance(indices, np.ndarray) or isinstance(indices, list):
            indices = np.asarray(indices)
            for i, j in enumerate(indices):
                if j < 0:
                    indices[i] += self.shape[0]
        elif isinstance(indices, slice):
            indices = np.asarray(tuple(range(*indices.indices(self.shape[0]))), dtype=np.int32)
        else:
            indices = np.asarray([indices], dtype=np.int32)
            if indices[0] < 0:
                indices[0] += self.x_shape[0]
        assert np.all(indices>=0) and np.all(indices<self.shape[0])

        result = np.empty([indices.shape[0], *self.shape[1:]], self.dtype)
        
        futures = []
        for i in range(result.shape[0]):            
            futures.append(core.thread_pool.submit(self.extract_patch, result[i], indices[i]))
        for future in futures:
            future.result()
            
        return result


class NormalizedAugmentedInputProvider(AbstractPatchProvider):
    def __init__(self, parent, channels, config):
        self.images = parent.base.images
        self.augmentation = parent.augmentation
        self.channels = channels
        self.ir_index = parent.base.ir_index
        self.red_index = parent.base.red_index
        self.normalization_params = parent.normalization_params
        self.num_classes = parent.num_classes
        self.extract_patch = self.extract_patch_numpy if isinstance(self.images[0].gt,np.ndarray) else self.extract_patch_sparse
    
        self.shape = (
            config.training_samples,
            self.channels.shape[0],
            *config.patch_size
        )
        self.dtype = np.float32
        
    def extract_patch_numpy(self, result, index):
        img = self.images[self.augmentation.image[index]]
        rust.extract_patch_bilinear_normalized(
            result, img.base, img.gt, img.depth,
            self.augmentation.transforms[index],
            self.augmentation.flips[index],
            self.augmentation.noise.seed[index],
            self.augmentation.noise.magnitude[index],
            self.channels, self.ir_index, self.red_index,
            self.normalization_params, self.num_classes
        )

    def extract_patch_sparse(self, result, index):
        img = self.images[self.augmentation.image[index]]
        rust.extract_patch_bilinear_normalized_sparse(
            result,
            img.base,
            img.gt.instances, img.gt.rows, img.gt.cols, img.gt.masks,
            img.depth,
            self.augmentation.transforms[index],
            self.augmentation.flips[index],
            self.augmentation.noise.seed[index],
            self.augmentation.noise.magnitude[index],
            self.channels, self.ir_index, self.red_index,
            self.normalization_params, self.num_classes
        )

        
class AugmentedInputProvider(AbstractPatchProvider):
    def __init__(self, parent, config):
        self.images = parent.base.images
        self.augmentation = parent.augmentation
        self.channels = parent.base.visualization_channels
        self.ir_index = parent.base.ir_index
        self.red_index = parent.base.red_index
        self.depth_range = parent.base.depth_range
        self.extract_patch = self.extract_patch_numpy if isinstance(self.images[0].gt,np.ndarray) else self.extract_patch_sparse
        
        self.shape = (
            config.training_samples,
            self.channels.shape[0],
            *config.patch_size
        )
        self.dtype = np.uint8
        
    def extract_patch_numpy(self, result, index):
        img = self.images[self.augmentation.image[index]]
        rust.extract_patch_bilinear(
            result, img.base, img.gt, img.depth,
            self.augmentation.transforms[index],
            self.augmentation.flips[index],
            self.channels, self.ir_index, self.red_index,
            self.depth_range
        )
        
    def extract_patch_sparse(self, result, index):
        img = self.images[self.augmentation.image[index]]
        rust.extract_patch_bilinear_sparse(
            result,
            img.base,
            img.gt.instances, img.gt.rows, img.gt.cols, img.gt.masks,
            img.depth,
            self.augmentation.transforms[index],
            self.augmentation.flips[index],
            self.channels, self.ir_index, self.red_index,
            self.depth_range
        )


class AugmentedOutputProvider(AbstractPatchProvider):
    def __init__(self, parent, config):
        self.images = parent.base.images
        self.augmentation = parent.augmentation
        self.extract_patch = self.extract_patch_numpy if isinstance(self.images[0].gt,np.ndarray) else self.extract_patch_sparse
        
        self.shape = (
            config.training_samples,
            *config.patch_size
        )
        self.dtype = np.int32
        
    def extract_patch_numpy(self, result, index):
        img = self.images[self.augmentation.image[index]]
        rust.extract_patch_nearest(
            result, img.gt,
            self.augmentation.transforms[index],
            self.augmentation.flips[index]
        )
        
    def extract_patch_sparse(self, result, index):
        img = self.images[self.augmentation.image[index]]
        rust.extract_patch_nearest_sparse(
            result,
            img.gt.instances, img.gt.rows, img.gt.cols, img.gt.masks, True,
            self.augmentation.transforms[index],
            self.augmentation.flips[index]
        )


class AugmentedInstanceProvider(AbstractPatchProvider):
    def __init__(self, parent, config):
        self.images = parent.base.images
        self.augmentation = parent.augmentation
        self.extract_patch = self.extract_patch_numpy if isinstance(self.images[0].gt,np.ndarray) else self.extract_patch_sparse
        
        self.shape = (
            config.training_samples,
            *config.patch_size
        )
        self.dtype = np.int32
        
    def extract_patch_numpy(self, result, index):
        img = self.images[self.augmentation.image[index]]
        rust.extract_patch_nearest(
            result, img.instances,
            self.augmentation.transforms[index],
            self.augmentation.flips[index]
        )
        
    def extract_patch_sparse(self, result, index):
        img = self.images[self.augmentation.image[index]]
        rust.extract_patch_nearest_sparse(
            result,
            img.instances.instances, img.instances.rows, img.instances.cols, img.instances.masks, False,
            self.augmentation.transforms[index],
            self.augmentation.flips[index]
        )


class NormalizedInputProvider(AbstractPatchProvider):
    def __init__(self, parent, patch_info, channels, patch_size):
        self.images = parent.base.images
        self.patch_info = patch_info
        self.channels = channels
        self.ir_index = parent.base.ir_index
        self.red_index = parent.base.red_index
        self.normalization_params = parent.normalization_params
        self.num_classes = parent.num_classes
        self.extract_patch = self.extract_patch_numpy if isinstance(self.images[0].gt,np.ndarray) else self.extract_patch_sparse
        
        self.shape = (
            self.patch_info.shape[0],
            self.channels.shape[0],
            *patch_size
        )
        self.dtype = np.float32
    
    def extract_patch_numpy(self, result, index):
        patch_info = self.patch_info[index]
        img = self.images[patch_info[0]]
        rust.get_patch_normalized(
            result, img.base, img.gt, img.depth, patch_info[1:],
            self.channels, self.ir_index, self.red_index,
            self.normalization_params, self.num_classes
        )
    
    def extract_patch_sparse(self, result, index):
        patch_info = self.patch_info[index]
        img = self.images[patch_info[0]]
        rust.get_patch_normalized_sparse(
            result,
            img.base,
            img.gt.instances, img.gt.rows, img.gt.cols, img.gt.masks,
            img.depth,
            patch_info[1:],
            self.channels, self.ir_index, self.red_index,
            self.normalization_params, self.num_classes
        )


class InputProvider(AbstractPatchProvider):
    def __init__(self, parent, patch_info, patch_size):
        self.images = parent.base.images
        self.patch_info = patch_info
        self.channels = parent.base.visualization_channels
        self.ir_index = parent.base.ir_index
        self.red_index = parent.base.red_index
        self.depth_range = parent.base.depth_range
        self.extract_patch = self.extract_patch_numpy if isinstance(self.images[0].gt,np.ndarray) else self.extract_patch_sparse
        
        self.shape = (
            self.patch_info.shape[0],
            self.channels.shape[0],
            *patch_size
        )
        self.dtype = np.uint8
        
    def extract_patch_numpy(self, result, index):
        patch_info = self.patch_info[index]
        img = self.images[patch_info[0]]
        rust.get_patch(
            result, img.base, img.gt, img.depth, patch_info[1:],
            self.channels, self.ir_index, self.red_index,
            self.depth_range
        )
        
    def extract_patch_sparse(self, result, index):
        patch_info = self.patch_info[index]
        img = self.images[patch_info[0]]
        rust.get_patch_sparse(
            result,
            img.base,
            img.gt.instances, img.gt.rows, img.gt.cols, img.gt.masks,
            img.depth,
            patch_info[1:],
            self.channels, self.ir_index, self.red_index,
            self.depth_range
        )


class OutputProvider(AbstractPatchProvider):
    def __init__(self, parent, patch_info, patch_size):
        self.images = parent.base.images
        self.patch_info = patch_info
        self.extract_patch = self.extract_patch_numpy if isinstance(self.images[0].gt,np.ndarray) else self.extract_patch_sparse
        
        self.shape = (
            self.patch_info.shape[0],
            *patch_size
        )
        self.dtype = np.int32
        
    def extract_patch_numpy(self, result, index):
        patch_info = self.patch_info[index]
        img = self.images[patch_info[0]]
        rust.get_gt_patch(
            result, img.gt, patch_info[1:]
        )
        
    def extract_patch_sparse(self, result, index):
        patch_info = self.patch_info[index]
        img = self.images[patch_info[0]]
        rust.get_gt_patch_sparse(
            result,
            img.gt.instances, img.gt.rows, img.gt.cols, img.gt.masks, True,
            patch_info[1:]
        )
        
        
class InstanceProvider(AbstractPatchProvider):
    def __init__(self, parent, patch_info, patch_size):
        self.images = parent.base.images
        self.patch_info = patch_info
        self.extract_patch = self.extract_patch_numpy if isinstance(self.images[0].gt,np.ndarray) else self.extract_patch_sparse
        
        self.shape = (
            self.patch_info.shape[0],
            *patch_size
        )
        self.dtype = np.int32
        
    def extract_patch_numpy(self, result, index):
        patch_info = self.patch_info[index]
        img = self.images[patch_info[0]]
        rust.get_gt_patch(
            result, img.instances, patch_info[1:]
        )
        
    def extract_patch_sparse(self, result, index):
        patch_info = self.patch_info[index]
        img = self.images[patch_info[0]]
        rust.get_gt_patch_sparse(
            result,
            img.instances.instances, img.instances.rows, img.instances.cols, img.instances.masks, False,
            patch_info[1:]
        )
