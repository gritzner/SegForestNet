import core
import datasets
import types
import numpy as np
import torch
from .SegmentationDatasetPatchProviders import AbstractPatchProvider


class ConcatenatedPatchProvider(AbstractPatchProvider):
    def __init__(self, parent, sources, update_image_ids):
        self.sources = sources
        self.shape = [0, *sources[0].shape[1:]]
        self.dtype = sources[0].dtype
        
        for source, offset in zip(sources, parent.offsets):
            source.no_threading = True
            source.images = parent.base.images
            if update_image_ids:
                if hasattr(source, "augmentation"):
                    source.augmentation.image += offset
                else:
                    source.patch_info[:,0] += offset
            self.shape[0] += source.shape[0]
            assert len(self.shape) == len(source.shape)
            for a, b in zip(self.shape[1:], source.shape[1:]):
                assert a == b
            assert self.dtype == source.dtype
            
        if hasattr(sources[0], "patch_info"):
            self.patch_info = np.concatenate([source.patch_info for source in sources], axis=0)
        
        self.source_ids = [], []
        offset = 0
        for i, source in enumerate(sources):
            self.source_ids[0].extend([i] * source.shape[0])
            self.source_ids[1].extend([offset] * source.shape[0])
            offset += source.shape[0]
        assert len(self.source_ids[0]) == self.shape[0]
        assert len(self.source_ids[0]) == len(self.source_ids[1])
        
    def extract_patch(self, result, index):
        i, j = self.source_ids[0][index], self.source_ids[1][index]
        result[:] = self.sources[i][index-j]


class InterleavedPatchProvider(AbstractPatchProvider):
    def __init__(self, parent, sources, update_image_ids):
        self.sources = sources
        self.shape = [0, *sources[0].shape[1:]]
        self.dtype = sources[0].dtype
        
        for source, offset in zip(sources, parent.offsets):
            source.no_threading = True
            source.images = parent.base.images
            if update_image_ids:
                if hasattr(source, "augmentation"):
                    source.augmentation.image += offset
                else:
                    source.patch_info[:,0] += offset
            self.shape[0] += source.shape[0]
            assert len(self.shape) == len(source.shape)
            for a, b in zip(self.shape[1:], source.shape[1:]):
                assert a == b
            assert self.dtype == source.dtype
            
        if hasattr(sources[0], "patch_info"):
            self.patch_info = np.concatenate([source.patch_info for source in sources], axis=0)
        
        self.source_ids = []
        for i, source in enumerate(sources):
            self.source_ids.extend(zip(
                [i]*source.shape[0], range(source.shape[0])
            ))
        self.source_ids.sort(key=lambda i: i[1])
        assert len(self.source_ids) == self.shape[0]
        
    def extract_patch(self, result, index):
        i, j = self.source_ids[index]
        result[:] = self.sources[i][j]


class MultiSegmentationDataset():
    def __init__(self, config, params):
        assert len(config.datasets) > 1
        all_datasets = [core.get_object_meta_info(dataset) for dataset in config.datasets]
        first = all_datasets[0][1]
        for _, other in all_datasets[1:]:
            assert len(first.channels.input) == len(other.channels.input)
            for a, b in zip(first.channels.input, other.channels.input):
                assert a == b
            assert len(first.patch_size) == len(other.patch_size)
            for a, b in zip(first.patch_size, other.patch_size):
                assert a == b
                
        all_datasets = [core.create_object(datasets, dataset) for dataset in config.datasets]
        first = all_datasets[0]
        assert not first.has_instances
        self.gsd = first.gsd
        self.base = types.SimpleNamespace(
            images = first.base.images,
            visualization_channels = first.base.visualization_channels
        )
        self.offsets = [0, len(self.base.images)]
        self.class_counts = first.class_counts
        for other in all_datasets[1:]:
            assert np.all(first.base.visualization_channels == other.base.visualization_channels)
            assert first.num_classes == other.num_classes
            assert first.ignore_class == other.ignore_class
            assert len(first.lut) == len(other.lut)
            for a, b in zip(first.lut, other.lut):
                assert len(a) == len(b)
                for c, d in zip(a, b):
                    assert c == d
            assert not other.has_instances
            self.gsd += other.gsd
            self.base.images.extend(other.base.images)
            self.offsets.append(len(self.base.images))
            self.class_counts += other.class_counts
        self.gsd /= len(all_datasets)
        
        self.num_classes = first.num_classes
        self.ignore_class = first.ignore_class
        self.lut = first.lut
        
        num_pixels = np.sum(self.class_counts)
        weights = np.empty(self.num_classes, dtype=np.float64)
        for c in range(self.num_classes):
            if self.class_counts[c] == num_pixels:
                weights[c] = 1
            elif c == self.ignore_class or self.class_counts[c] == 0:
                weights[c] = 0
            else:
                weights[c] = 1 - self.class_counts[c]/num_pixels
        weights = weights / np.max(weights)
        self.class_weights = torch.from_numpy(weights).float().to(core.device)
        
        patch_provider = InterleavedPatchProvider if config.interleave else ConcatenatedPatchProvider
        self.training = types.SimpleNamespace(
            x = patch_provider(self, [dataset.training.x for dataset in all_datasets], True),
            x_gt = patch_provider(self, [dataset.training.x_gt for dataset in all_datasets], False),
            x_vis = patch_provider(self, [dataset.training.x_vis for dataset in all_datasets], False),
            y = patch_provider(self, [dataset.training.y for dataset in all_datasets], False)
        )
        self.validation = types.SimpleNamespace(
            x = patch_provider(self, [dataset.validation.x for dataset in all_datasets], True),
            x_gt = patch_provider(self, [dataset.validation.x_gt for dataset in all_datasets], False),
            x_vis = patch_provider(self, [dataset.validation.x_vis for dataset in all_datasets], False),
            y = patch_provider(self, [dataset.validation.y for dataset in all_datasets], False),
            index_map = patch_provider(self, [dataset.validation.index_map for dataset in all_datasets], False)
        )
        self.test = types.SimpleNamespace(
            x = patch_provider(self, [dataset.test.x for dataset in all_datasets], True),
            x_gt = patch_provider(self, [dataset.test.x_gt for dataset in all_datasets], False),
            x_vis = patch_provider(self, [dataset.test.x_vis for dataset in all_datasets], False),
            y = patch_provider(self, [dataset.test.y for dataset in all_datasets], False),
            index_map = patch_provider(self, [dataset.test.index_map for dataset in all_datasets], False)
        )
        
        del self.offsets
