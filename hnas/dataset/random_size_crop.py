import random
import paddle.vision.transforms as F

from paddle.vision.transforms import RandomResizedCrop


class MyRandomResizedCrop(RandomResizedCrop):
    current_size = 224
    image_size_list = [112, 160, 192, 224]

    epoch = 0
    batch_id = 0

    def __init__(self, size_list, scale=(0.08, 1.0), ratio=(3. / 4, 4. / 3), interpolation='bilinear', keys=None):
        assert isinstance(size_list, list)
        MyRandomResizedCrop.image_size_list = size_list

        super(MyRandomResizedCrop, self).__init__(MyRandomResizedCrop.image_size_list[0], scale, ratio, interpolation)

    def _apply_image(self, img):
        i, j, h, w = self._get_param(img)

        cropped_img = F.crop(img, i, j, h, w)
        return F.resize(cropped_img, 
                        (MyRandomResizedCrop.current_size, MyRandomResizedCrop.current_size), 
                        self.interpolation)

    @staticmethod
    def sample_image_size(batch_id):
        _seed = int('%d%.3d' % (batch_id, MyRandomResizedCrop.epoch))
        random.seed(_seed)
        MyRandomResizedCrop.current_size = random.choices(MyRandomResizedCrop.image_size_list)[0]