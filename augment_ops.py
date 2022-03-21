from PIL import Image
from PIL import ImageEnhance, ImageDraw
from datetime import datetime
import random
import numpy as np
import math


from process import save_image_bbox, parse_bbox_xml

__all__ = [
    "RandomImageEnhance",
    "RandomNoise",
    "RandomFlip",
    "RandomTranslate",
    "RandomRotate"
]

def _clip_bbox(bbox_change, bbox_org, image_shape, overlap=0.6):
    """Clip bbox into image
    保证变换后的bbox在图片范围内，且变换后的bbox的面积至少大于overlap*变换前的bbox的面积

    Parameter:
        bbox_change: list 变换后的bbox [[x1, y1, x2, y2], ...]
        bbox_org: list 变换前的bbox [[x1, y1, x2, y2], ...]
        image_shape: tuple 图片大小 (w,h)
        overlap: float(0~1)变换前后bbox的重叠率
    
    Output:
        bbox_out: bbox是否符合要求，若不符合要求则丢弃当前变换

    """
    bbox_out = []
    for i in range(len(bbox_change)):
        if bbox_change[i][0] < 0:
            bbox_change[i][0] = 0
        elif bbox_change[i][0] > image_shape[0]:
            bbox_change[i][0] = image_shape[0]
        
        if bbox_change[i][2] < 0:
            bbox_change[i][2] = 0
        elif bbox_change[i][2] > image_shape[0]:
            bbox_change[i][2] = image_shape[0]

        if bbox_change[i][1] < 0:
            bbox_change[i][1] = 0
        elif bbox_change[i][1] > image_shape[1]:
            bbox_change[i][1] = image_shape[1]

        if bbox_change[i][3] < 0:
            bbox_change[i][3] = 0
        elif bbox_change[i][3] > image_shape[1]:
            bbox_change[i][3] = image_shape[1]

        bbox_change_w = abs(bbox_change[i][2] - bbox_change[i][0])
        bbox_change_h = abs(bbox_change[i][3] - bbox_change[i][1])
        bbox_change_size = (bbox_change_w * bbox_change_h)

        bbox_org_w = abs(bbox_org[i][2] - bbox_org[i][0])
        bbox_org_h = abs(bbox_org[i][3] - bbox_org[i][1])
        bbox_org_size = (bbox_org_w * bbox_org_h)
        # print("size of bbox changed: {}, size of bbox original: {}".format(bbox_change_size, bbox_org_size * overlap))
        # if bbox_change_size >= bbox_org_size * overlap:
        #    bbox_out.append(bbox_change[i])

    return bbox_change

def _get_rotation_matrix(center, angle, scale):
    """Calculate rotation matrix

    Parameter:
        center: image center (cx, cy)
        angle: degree of angle
        scale
    """
    angle = math.radians(angle)
    alpha = math.cos(angle) * scale
    beta = math.sin(angle) * scale

    matrix = np.zeros((2, 3))
    matrix[0, 0] = alpha
    matrix[0, 1] = beta
    matrix[0, 2] = (1.0 - alpha) * center[0] - beta * center[1]
    matrix[1, 0] = -beta
    matrix[1, 1] = alpha
    matrix[1, 2] = beta * center[0] + (1 - alpha) * center[1]

    return matrix


class RandomImageEnhance(object):
    """Randomly change the image's contrast, color saturation, sharpness and brightness.

    Parameters: 
        contrast_ratio_range: 对比度的变化范围
        color_coeff_range: 颜色增强系数的变化范围
        sharpness_range: 锐度变化范围
        brightness_range: 亮度变化范围

    Input:
        image: PIL format image class
        bbox: list of bounding boxes

    Usage:
        Initialize class: enhance_util = RandomImageEnhance(...)
        Call for generating augmented images: enhance_util(image, bbox)
        Save the generated images: enhance_util.save()
    """
    def __init__(self, contrast_ratio_range=(1, 5), color_coeff_range=(0.5, 2.5), sharpness_range=(5, 20), brightness_range=(0.5, 2.5)):
        self._contrast_ratio_range = contrast_ratio_range
        self._color_coeff_range = color_coeff_range
        self._sharpness_range = sharpness_range
        self._brightness_range = brightness_range
        self._enhance_options = ['color', 'sharpness', 'brightness']


    def __call__(self, image, bbox):
        contrast_ratio = random.uniform(*self._contrast_ratio_range)
        image_enhanced = ImageEnhance.Contrast(image).enhance(contrast_ratio)
        operation_num = random.randint(0,2)
        other_operations = random.sample(self._enhance_options, operation_num)
        if 'color' in other_operations:
            color_coeff = random.uniform(*self._color_coeff_range)
            image_enhanced = ImageEnhance.Color(image_enhanced).enhance(color_coeff)
        if 'sharpness' in other_operations:
            sharp_ratio = random.uniform(*self._sharpness_range)
            image_enhanced = ImageEnhance.Sharpness(image_enhanced).enhance(sharp_ratio)
        if 'brightness' in other_operations:
            brightness_ratio = random.uniform(*self._brightness_range)
            image_enhanced = ImageEnhance.Brightness(image_enhanced).enhance(brightness_ratio)
        
        self._save_image = image_enhanced
        self._bbox = bbox
        self._contrast_ratio = contrast_ratio
    
    def save(self, DOM, save_image_dir, save_anno_dir, image_name):
        if image_name.endswith('.jpeg'):
            name = image_name[:-5]
            suffix = '.jpeg'
        else:
            name = image_name[:-4]
            if image_name.endswith('.jpg'):
                suffix = '.jpg'
            elif image_name.endswith('.png'):
                suffix = '.png'
        now_time_stamp = str(datetime.now())[11:-7]
        new_image_name = name + '_enhance' + '_' + str(self._contrast_ratio) + '_' + now_time_stamp + suffix
        bbox_xml_name = name + '_enhance' + '_' + str(self._contrast_ratio) + '_' + now_time_stamp + '.xml'
        save_image_bbox(DOM, save_image_dir, save_anno_dir, new_image_name, bbox_xml_name, self._save_image, self._bbox)

    def draw(self):
        drawer = ImageDraw.Draw(self._save_image)
        for i in range(len(self._bbox)):
            drawer.rectangle(self._bbox[i], outline=(255,0,0), width=3)
        self._save_image.show()  


class RandomNoise(object):
    """Randomly add noise into image.

    Parameters:
        pepper_prob: 椒盐噪声的比例
        gaussian_mean: 高斯噪声的均值
        gaussian_var: 高斯噪声的方差
        noise_type: 噪声种类
    
    Input:
        image: PIL.Image
        bbox: list of bounding boxes
    """

    def __init__(self, noise_type='gaussian', gaussian_mean=0, gaussian_var=0.005, pepper_prob=0.05):
        self._noise_type = noise_type
        self._gaussian_mean = gaussian_mean
        self._gaussian_var = gaussian_var
        self._pepper_prob = pepper_prob

    def __call__(self, image, bbox):
        if self._noise_type == 'gaussian':
            self._gaussian_noise(image, bbox)
        elif self._noise_type == 'pepper_noise':
            self._pepper_noise(image, bbox)
    
    def _gaussian_noise(self, image, bbox):
        image_np = np.asarray(image, dtype=np.float) / 255.0
        noise = np.random.normal(self._gaussian_mean, self._gaussian_var ** 0.5, image_np.shape)
        image_out = image_np + noise
        if image_out.min() < 0:
            low_clip = -1.0
        else:
            low_clip = 0.0
        image_out = np.clip(image_out, low_clip, 1.0)
        image_out = np.uint8(image_out * 255.0)
        self._save_image = Image.fromarray(image_out)
        self._bbox = bbox

    def _pepper_noise(self, image, bbox):
        image_np = np.asarray(image)
        image_out = np.zeros(image_np.shape, dtype=np.uint8)
        thres = 1 - self._pepper_prob
        for i in range(image_np.shape[0]):
            for j in range(image_np.shape[1]):
                rand_prob = random.random()
                if rand_prob < self._pepper_prob:
                    image_out[i, j, :] = 0
                elif rand_prob > thres:
                    image_out[i, j, :] = 255
                else:
                    image_out[i, j, :] = image_np[i, j, :]
        self._save_image = Image.fromarray(image_out)
        self._bbox = bbox

    def save(self, DOM, save_image_dir, save_anno_dir, image_name):
        if image_name.endswith('.jpeg'):
            name = image_name[:-5]
            suffix = '.jpeg'
        else:
            name = image_name[:-4]
            if image_name.endswith('.jpg'):
                suffix = '.jpg'
            elif image_name.endswith('.png'):
                suffix = '.png'
        now_time_stamp = str(datetime.now())[11:-7]
        new_image_name = name + '_noise' + '_' + now_time_stamp + suffix
        bbox_xml_name = name + '_noise' + '_' + now_time_stamp + '.xml'
        save_image_bbox(DOM, save_image_dir, save_anno_dir, new_image_name, bbox_xml_name, self._save_image, self._bbox)

    def draw(self):
        drawer = ImageDraw.Draw(self._save_image)
        for i in range(len(self._bbox)):
            drawer.rectangle(self._bbox[i], outline=(255,0,0), width=3)
        self._save_image.show()   

################################################################

class RandomFlip(object):
    """Randomly flip image, including vertical flip, horizontal flip

    Parameter:
        prob: probability of vertical or horizontal flipping
    
    Input:
        image: PIL.Image
        bbox: list of bounding boxes  [[x1, y1, x2, y2], ...]
    """
    def __init__(self, prob=0.5):
        self._prob = prob

    def __call__(self, image, bbox):
        center = np.uint8(np.array(image.size) / 2)  # (w/2,h/2)
        center = np.hstack((center, center))
        bbox_np = np.array(bbox)
        
        if random.random() < self._prob:
            # horizontal flip
            image_out = image.transpose(Image.FLIP_LEFT_RIGHT)
            bbox_np[:, [0, 2]] += 2 * (center[[0, 2]] - bbox_np[:, [0, 2]])  # flip x coordinate
            box_w = abs(bbox_np[:, 0] - bbox_np[:, 2])
            bbox_np[:, 0] -= box_w
            bbox_np[:, 2] += box_w
        else:
            # vertical flip
            image_out = image.transpose(Image.FLIP_TOP_BOTTOM)
            bbox_np[:, [1, 3]] += 2 * (center[[1, 3]] - bbox_np[:, [1, 3]])  # flip y coordinate
            box_h = abs(bbox_np[:, 1] - bbox_np[:, 3])
            bbox_np[:, 1] -= box_h
            bbox_np[:, 3] += box_h

        self._save_image = image_out
        self._bbox = bbox_np.tolist()

    def save(self, DOM, save_image_dir, save_anno_dir, image_name):
        if image_name.endswith('.jpeg'):
            name = image_name[:-5]
            suffix = '.jpeg'
        else:
            name = image_name[:-4]
            if image_name.endswith('.jpg'):
                suffix = '.jpg'
            elif image_name.endswith('.png'):
                suffix = '.png'
        now_time_stamp = str(datetime.now())[11:-7]
        new_image_name = name + '_flip' + '_' + now_time_stamp + suffix
        bbox_xml_name = name + '_flip' + '_' + now_time_stamp + '.xml'
        save_image_bbox(DOM, save_image_dir, save_anno_dir, new_image_name, bbox_xml_name, self._save_image, self._bbox)

    def draw(self):
        drawer = ImageDraw.Draw(self._save_image)
        for i in range(len(self._bbox)):
            drawer.rectangle(self._bbox[i], outline=(255,0,0), width=3)
        self._save_image.show()         

#######################################################
class RandomTranslate(object):
    """Randomly Translate the images

    Parameters:
        translate_ratio_range: 平移系数的变化范围，(0~1)；乘上长宽得到平移的长度；float类型

    Input:
        image: PIL.Image
        bbox: list of bounding boxes  [[x1, y1, x2, y2], ...]
    """
    def __init__(self, translate_ratio_range=0.2, diff=False):
        assert translate_ratio_range > 0 and translate_ratio_range < 1, "ratio must be between 0 and 1."
        if diff:
            self._translate_ratio_x = random.uniform(-translate_ratio_range, translate_ratio_range)
            self._translate_ratio_y = random.uniform(-translate_ratio_range, translate_ratio_range)
        else:
            random_ratio = random.uniform(-translate_ratio_range, translate_ratio_range)
            self._translate_ratio_x = random_ratio
            self._translate_ratio_y = random_ratio
            
        
    def __call__(self, image, bbox):
        w, h = image.size 
        corner_x = int(self._translate_ratio_x * w)
        corner_y = int(self._translate_ratio_y * h)
        image_np = np.asarray(image)   # w h c
        image_out = np.zeros(image_np.shape).astype(np.uint8)

        translated_cord = [max(0, corner_y), max(0, corner_x), min(h, corner_y + h), min(w, corner_x + w)]
        mask = image_np[max(0, -corner_y):min(h, h - corner_y), max(0, -corner_x):min(w, w - corner_x),:]
        image_out[translated_cord[0]:translated_cord[2], translated_cord[1]:translated_cord[3],:] = mask
        
        bbox_np = np.array(bbox)
        bbox_np[:,:4] += [corner_x, corner_y, corner_x, corner_y]
        bbox_out = _clip_bbox(bbox_np.tolist(), bbox, (w, h))
        self._save_image = Image.fromarray(image_out)
        self._bbox = bbox_out


    def save(self, DOM, save_image_dir, save_anno_dir, image_name):
        if image_name.endswith('.jpeg'):
            name = image_name[:-5]
            suffix = '.jpeg'
        else:
            name = image_name[:-4]
            if image_name.endswith('.jpg'):
                suffix = '.jpg'
            elif image_name.endswith('.png'):
                suffix = '.png'
        now_time_stamp = str(datetime.now())[11:-7]
        new_image_name = name + '_translate'  + '_' + now_time_stamp + suffix
        bbox_xml_name = name + '_translate' +  '_' + now_time_stamp + '.xml'
        save_image_bbox(DOM, save_image_dir, save_anno_dir, new_image_name, bbox_xml_name, self._save_image, self._bbox)

    def draw(self):
        drawer = ImageDraw.Draw(self._save_image)
        for i in range(len(self._bbox)):
            drawer.rectangle(self._bbox[i], outline=(255,0,0), width=3)
        self._save_image.show()  

class RandomRotate(object):
    """Randomly Rotate the image

    Parameters:
        angle_range: 旋转角度的变化范围 tuple (-180 ~ 180)
    
    Input:
        image: PIL.Image
        bbox: list of bounding boxes
    """
    def __init__(self, angle_range=(-20, 20)):
        self._angle_range = angle_range
        assert angle_range[0] > -180 and angle_range[1] < 180, "Angle range is invalid."
    
    def __call__(self, image, bbox):
        w, h = image.size
        cx, cy = (w // 2, h // 2)

        random_angle = int(random.uniform(*self._angle_range))
        if random_angle == 0:
            random_angle = random.randint(1,10)
        image_rotated = image.rotate(random_angle,expand=1)  # 直接调用Image的旋转图像函数，_rotate_image有点bug待解决
        bbox_rotated = self._rotate_bbox(bbox, random_angle, (cx, cy), (w, h))

        # scale rotated image to original size
        scale_x = image_rotated.size[0] / w
        scale_y = image_rotated.size[1] / h

        image_rotated_resize = image_rotated.resize((w, h))
        bbox_rotated[:,:4] /= [scale_x, scale_y, scale_x, scale_y]
        bbox_rotated = np.int16(bbox_rotated)
        self._save_image = image_rotated_resize
        self._bbox = bbox_rotated.tolist()
        self._rotate_angle = random_angle

    def save(self, DOM, save_image_dir, save_anno_dir, image_name):
        if image_name.endswith('.jpeg'):
            name = image_name[:-5]
            suffix = '.jpeg'
        else:
            name = image_name[:-4]
            if image_name.endswith('.jpg'):
                suffix = '.jpg'
            elif image_name.endswith('.png'):
                suffix = '.png'
        now_time_stamp = str(datetime.now())[11:-7]
        new_image_name = name + '_rotate' + '_' + str(self._rotate_angle) + '_' + now_time_stamp + suffix
        bbox_xml_name = name + '_rotate' + '_' + str(self._rotate_angle) + '_' + now_time_stamp + '.xml'
        save_image_bbox(DOM, save_image_dir, save_anno_dir, new_image_name, bbox_xml_name, self._save_image, self._bbox)

    def draw(self):
        drawer = ImageDraw.Draw(self._save_image)
        for i in range(len(self._bbox)):
            drawer.rectangle(self._bbox[i], outline=(255,0,0), width=3)
        # self._save_image.show()
        return self._save_image
    
    def _rotate_image(self, image, angle):
        """Rotate a image with a given angle

        Parameters:
            image: PIL.Image object
            angle: float
        """
        w, h = image.size
        cx, cy = (w // 2, h // 2)

        rotated_matrix = _get_rotation_matrix((cx, cy), angle, 1.0)
        cos = np.abs(rotated_matrix[0, 0])
        sin = np.abs(rotated_matrix[0, 1])

        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) - (w * sin))

        rotated_matrix[0, 2] += (new_w / 2) - cx
        rotated_matrix[1, 2] += (new_h / 2) - cy
        # TODO: 旋转矩阵的参数顺序与Image.transform的格式不一样
        rotated_matrix_tuple = tuple(rotated_matrix.reshape(-1).tolist())

        image_rotated = image.copy()
        image_rotated = image_rotated.transform((new_w, new_h), Image.AFFINE, rotated_matrix_tuple)
        return image_rotated

    def _rotate_bbox(self, bbox, angle, image_center, image_shape):
        """Rotate bbox with a given angle

        Parameters:
            bbox: list of bounding boxes
            angle: float angle
            image_center: tuple
            image_shape: tuple
        
        Output:
            bbox_np: np.array
        """
        def _get_corners():
            w = (bbox_np[:, 2] - bbox_np[:, 0]).reshape(-1, 1)
            h = (bbox_np[:, 3] - bbox_np[:, 1]).reshape(-1, 1)
            # corner one
            x1 = bbox_np[:, 0].reshape(-1, 1)
            y1 = bbox_np[:, 1].reshape(-1, 1)
            # corner two
            x2 = x1 + w
            y2 = y1
            # corner three
            x3 = x1
            y3 = y1 + h
            #corner four
            x4 = bbox_np[:, 2].reshape(-1, 1)
            y4 = bbox_np[:, 3].reshape(-1, 1)
            
            corners = np.hstack((x1, y1, x2, y2, x3, y3, x4, y4))

            return corners

        def _postprocess_bbox():
            """postprocess the bbox generated from rotation

            """
            xs_ = out[:, [0, 2, 4, 6]]
            ys_ = out[:, [1, 3, 5, 7]]

            x_min = np.min(xs_, 1).reshape(-1, 1)
            y_min = np.min(ys_, 1).reshape(-1, 1)
            x_max = np.max(xs_, 1).reshape(-1, 1)
            y_max = np.max(ys_, 1).reshape(-1, 1)

            return np.hstack((x_min, y_min, x_max, y_max))

        bbox_np = np.array(bbox)
        corners = _get_corners().reshape(-1, 2)
        corners = np.hstack((corners, np.ones((corners.shape[0], 1), dtype=type(corners[0][0]))))

        cx, cy = image_center
        w, h = image_shape
        rotated_matrix = _get_rotation_matrix((cx, cy), angle, 1.0)
        cos = np.abs(rotated_matrix[0, 0])
        sin = np.abs(rotated_matrix[0, 1])

        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        rotated_matrix[0, 2] += (new_w / 2) - cx
        rotated_matrix[1, 2] += (new_h / 2) - cy

        out = np.dot(rotated_matrix, corners.T).T
        out = out.reshape(-1, 8)
        bbox_out = _postprocess_bbox()
        return bbox_out

if __name__ == "__main__":
    image = Image.open('test2/image_0.jpg')
    DOM, bboxes = parse_bbox_xml('test2', 'image_0.xml')
    test = RandomFlip()
    test(image, bboxes)
    test.save(DOM, 'test2', 'test2', 'image_0.jpg')
    test.draw()

