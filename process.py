from xml.dom.minidom import parse
import os

__all__ = [
    "parse_bbox_xml",
    "save_image_bbox"
]

def parse_bbox_xml(file_dir, xml_name):
    """Parse bbox from xml file

    Parameters:
        file_dir: 标注xml文件所在文件夹
        xml_name: 文件名
    
    Return:
        domTree: xml的DOM对象
        bboxes: list of bounding boxes
    """
    domTree = parse(os.path.join(file_dir, xml_name))
    rootNode = domTree.documentElement

    bboxes = []
    bbox_objects = rootNode.getElementsByTagName('object')
    # debug
    for bbox_obj in bbox_objects:
        x1 = int(bbox_obj.childNodes[9].childNodes[1].childNodes[0].data)
        y1 = int(bbox_obj.childNodes[9].childNodes[3].childNodes[0].data)
        x2 = int(bbox_obj.childNodes[9].childNodes[5].childNodes[0].data)
        y2 = int(bbox_obj.childNodes[9].childNodes[7].childNodes[0].data)
        bboxes.append([x1, y1, x2, y2])
    
    return domTree, bboxes

def save_image_bbox(DOM, save_image_dir, save_anno_dir, image_filename, xml_name, image, bboxes):
    """Save image and bbox to xml file

    Parameter: 
        DOM: DOM对象
        save_dir: directory to save
        image_filename: filename of save image
        bbox_xmlname: xml name of save bboxes
        image: PIL.Image object
        bboxes: list of bounding boxes
    """
    rootNode = DOM.documentElement
    # update image filename
    filename_node = rootNode.getElementsByTagName('filename')[0]
    filename_node.childNodes[0].data = image_filename
    # update image path
    image_path = os.path.join(save_image_dir, image_filename)
    path_node = rootNode.getElementsByTagName('path')[0]
    path_node.childNodes[0].data = image_path
    # update bboxes
    bbox_objects = rootNode.getElementsByTagName('object')
    if len(bbox_objects) != len(bboxes):
        print('There are some annotations might be discarded. The number of discarded bboxes is {}'.format(len(bbox_objects) - len(bboxes)))
    # assert len(bbox_objects) == len(bboxes), "length of bboxes must be equal to the number of xml nodes."
    for i, bbox_obj in enumerate(bbox_objects):
        bbox_obj.childNodes[9].childNodes[1].childNodes[0].data = str(bboxes[i][0])  # xmin
        bbox_obj.childNodes[9].childNodes[3].childNodes[0].data = str(bboxes[i][1])  # ymin
        bbox_obj.childNodes[9].childNodes[5].childNodes[0].data = str(bboxes[i][2])  # xmax
        bbox_obj.childNodes[9].childNodes[7].childNodes[0].data = str(bboxes[i][3])  # ymax

    # write to xml file 
    xml_path = os.path.join(save_anno_dir, xml_name)
    with open(xml_path, 'w') as f:
        DOM.writexml(f, encoding='utf-8')
    
    # save image
    image.save(image_path)



