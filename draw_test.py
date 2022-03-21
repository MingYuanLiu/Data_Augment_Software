from PIL import Image
from augment_ops import RandomFlip, RandomTranslate, RandomImageEnhance
from augment_ops import RandomNoise, RandomRotate
from process import parse_bbox_xml

import os

def draw_test(image_path, xml_path, save_dir='./draw'):
    """
    使用该程序画出变换后的图像和标注框
    """
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    image = Image.open(image_path)
    xml_dir = os.path.dirname(xml_path)
    xml_filename = xml_path.split('/')[-1]
    DOM, bboxes = parse_bbox_xml(xml_dir, xml_filename)
    test = RandomRotate()
    test(image, bboxes)
    test.save(DOM, save_dir, save_dir, image_path.split('/')[-1])
    draw_image = test.draw()
    return draw_image

if __name__ == "__main__":
    draw_test('test/image_4_rotate_10_09:23:05.jpg', 'test/image_4_rotate_10_09:23:05.xml', 'test')
