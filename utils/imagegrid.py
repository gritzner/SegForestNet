import numpy as np
import PIL.Image


class ImageGrid():
    def __init__(self, patch_size, size, channels_first=True, border=0, margins=(0,0), bg_color=(127,127,127)):
        assert len(patch_size) == 2 and patch_size[0] > 0 and patch_size[1] > 0
        assert len(size) == 2 and size[0] > 0 and size[1] > 0

        self.patch_size = tuple(patch_size)
        self.size = tuple(size)
        self.channels_first = channels_first
        self.border = border
        
        shape = [2*border + patch_size[i]*size[i] + margins[i]*(size[i]-1) for i in range(2)]
        self.image = np.empty((*shape, 3), dtype=np.uint8)
        self.image[:] = np.expand_dims(np.expand_dims(np.asarray(bg_color), axis=0), axis=0)
        self.strides = tuple([patch_size[i]+margins[i] for i in range(2)])
    
    def set_image(self, i, j, image, lut=None):
        assert 0 <= i < self.size[0]
        assert 0 <= j < self.size[1]
        if len(image.shape) > 2:
            if self.channels_first:
                image = np.moveaxis(image, 0, -1)
        elif lut:
            image = ImageGrid.apply_lut(image, lut)
        else:
            image = np.expand_dims(image, axis=-1)
        image = np.asarray(np.clip(image, 0, 255), dtype=np.uint8)
        if image.shape[0] < self.patch_size[0] or image.shape[1] < self.patch_size[1]:
            v_repetitions = self._get_repetitions(image, 0)
            h_repetitions = self._get_repetitions(image, 1)
            image = np.repeat(np.repeat(image, h_repetitions, axis=1), v_repetitions, axis=0)
        v_pos = self.border + i * self.strides[0]
        h_pos = self.border + j * self.strides[1]
        self.image[v_pos:v_pos+self.patch_size[0],h_pos:h_pos+self.patch_size[1]] = image
        
    @staticmethod
    def apply_lut(src_img, lut):
        dst_img = np.empty([*src_img.shape, 3], dtype=np.uint8)
        for i, rgb in enumerate(lut):
            j = np.where(src_img==i)
            dst_img[j[0],j[1]] = rgb
        return dst_img
    
    def _get_repetitions(self, image, index):
        assert self.patch_size[index] % image.shape[index] == 0
        return self.patch_size[index] // image.shape[index]
    
    def save(self, filename):
        PIL.Image.fromarray(self.image).save(filename)
        