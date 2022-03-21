from PIL import Image
from augment_ops import RandomFlip, RandomTranslate, RandomImageEnhance
from augment_ops import RandomNoise, RandomRotate
from process import parse_bbox_xml

import random
import os
from multiprocessing.pool import Pool
from tqdm import tqdm

def thread_process(params):
    """Thread Process
    
    Parameter:
        params[0]: image_dir
        params[1]: xml_dir
        params[2]: save_image_dir
        params[3]: save_xml_dir
        params[4]: image_files
        params[5]: anno_xml_files
    """
    image_dir = params[0]
    xml_dir = params[1]

    save_image_dir = params[2]
    save_xml_dir = params[3]

    image_filename = params[4]
    anno_xml_filename = params[5]
    # parse xml
    DOM, bboxes = parse_bbox_xml(xml_dir, anno_xml_filename)
    image = Image.open(os.path.join(image_dir, image_filename))

    # select augment types
    augment_types_all = ['flip', 'translate', 'enhance', 'noise', 'rotate']
    random_sel_types = random.sample(augment_types_all, 3)

    for aug_type in random_sel_types:
        if aug_type == 'flip':
            augmentor = RandomFlip()
        elif aug_type == 'translate':
            augmentor = RandomTranslate()
        elif aug_type == 'enhance':
            augmentor = RandomImageEnhance()
        elif aug_type == 'noise':
            augmentor = RandomNoise()
        elif aug_type == 'rotate':
            augmentor = RandomRotate()
        augmentor(image, bboxes)
        augmentor.save(DOM, save_image_dir, save_xml_dir, image_filename)


def main(image_dir, xml_dir, save_image_dir, save_xml_dir):
    image_files = [x for x in os.listdir(image_dir) if x.endswith('.jpg') or x.endswith('.png') or x.endswith('.jpeg')]
    xml_files = [x for x in os.listdir(xml_dir) if x.endswith('.xml')]
    print(xml_dir, image_dir, save_image_dir, save_xml_dir)
    # assert len(image_files) == len(xml_files), "Image numbers is not equal to annotation files. Please check!"
    thread_args = []
    for i, image_filename in enumerate(image_files):
        if image_filename.endswith('.jpg') or image_filename.endswith('.png'):
            xml_filename = image_filename[:-4] + '.xml'
        else:
            xml_filename = image_filename[:-5] + '.xml'
        if xml_filename in xml_files:
            arg_temp = [image_dir, xml_dir, save_image_dir, save_xml_dir, image_filename, xml_filename]
            thread_args.append(arg_temp)

    with Pool(processes=1) as p:
        res = list(tqdm(p.imap(thread_process, thread_args), total=len(thread_args), desc='Running data augmentation ...'))
    p.close()
    p.join()

    print('Data augmentation finished')
    return True

import tkinter as tk
from tkinter.filedialog import askdirectory, askopenfilename
from PIL import ImageTk
from draw_test import draw_test

class App(object):
    """Interactivate interface

    """
    def __init__(self):
        self._window = tk.Tk()
        self._window.geometry('840x600')
        self._window.title('Demo - Offine data augmentation for object detection. Author: Mingyuan Liu')

        self._selected_image_dir = tk.StringVar()
        self._selected_xml_dir = tk.StringVar()
        self._save_image_dir = tk.StringVar()
        self._save_xml_dir = tk.StringVar()
        self._finished = tk.StringVar()

        self._image_tklabel = None
        self._draw_image_path_var = tk.StringVar()
        self._draw_xml_path_var = tk.StringVar()
        self._cover_image_path = 'image/result.png'

        self.layout()
    
    def layout(self):
        tk.Label(self._window, text='Data augmentation', font=('Atril', 16)).grid(row=0, column=0)
        tk.Label(self._window, text='--------------------------').grid(row=1, column=0)
        tk.Label(self._window, text='Image directory: ', font=('Atril', 12)).grid(row=2, column=0)
        tk.Entry(self._window, textvariable=self._selected_image_dir, width=45, bd=3).grid(row=2, column=1)
        tk.Button(self._window, text='select image dir', command=self.select_image_dir, font=('Atril', 8)).grid(row=2, column=2, stick=tk.W)

        tk.Label(self._window, text='Annotation directory: ', font=('Atril', 12)).grid(row=3, column=0)
        tk.Entry(self._window, textvariable=self._selected_xml_dir, width=45, bd=3).grid(row=3, column=1)
        tk.Button(self._window, text='select xml dir', command=self.select_xml_dir, font=('Atril', 8)).grid(row=3, column=2, stick=tk.W)

        tk.Label(self._window, text='Save image directory: ', font=('Atril', 12)).grid(row=4, column=0)
        tk.Entry(self._window, textvariable=self._save_image_dir, width=45, bd=3).grid(row=4, column=1, sticky=tk.NW)
        tk.Button(self._window, text='select save image dir', command=self.select_save_image_dir, font=('Atril', 8)).grid(row=4, column=2, sticky=tk.W)
        
        tk.Label(self._window, text='Save xml directory: ', font=('Atril', 12)).grid(row=5, column=0)
        tk.Entry(self._window, textvariable=self._save_xml_dir, width=45, bd=3).grid(row=5, column=1)
        tk.Button(self._window, text='select save xml dir', command=self.select_save_xml_dir, font=('Atril', 8)).grid(row=5, column=2, sticky=tk.W)
        tk.Label(self._window, text='', font=('Atril', 16)).grid(row=6, column=0)
        tk.Button(self._window, text='Data augmentation', command=self.run_data_augment, font=('Atril', 12)).grid(row=7, column=2, sticky=tk.W)
        tk.Label(self._window, textvariable=self._finished, font=('Atril', 12), bg='Blue').grid(row=7,column=1)

        # Draw picture
        tk.Label(self._window, text='', font=('Atril', 16)).grid(row=8, column=0, columnspan=2)
        tk.Label(self._window, text='Draw augmentation picture', font=('Atril', 16)).grid(row=9, column=0)
        tk.Label(self._window, text='----------------------------------').grid(row=10, column=0)
        cover_image_pil = Image.open(self._cover_image_path).resize((280, 240))
        cover_image_tk = ImageTk.PhotoImage(cover_image_pil)
        self._image_tklabel = tk.Label(self._window, image=cover_image_tk, width=280, height=240)
        self._image_tklabel.grid(row=11,column=0)
        # tk.Label(self._window, text='Image path: ', font=('Atril', 16)).grid(row=17, column=1,  sticky=tk.NW)
        tk.Entry(self._window, textvariable=self._draw_image_path_var, width=40, bd=3).grid(row=11, column=1, sticky=tk.N)
        tk.Button(self._window, text='select image path ', command=self.select_draw_image_path, font=('Atril', 8)).grid(row=11, column=2, sticky=tk.NW)
        # tk.Label(self._window, text='XML_path: ', font=('Atril', 16)).grid(row=53, column=0, columnspan=2)
        tk.Entry(self._window, textvariable=self._draw_xml_path_var, width=40, bd=3).grid(row=12, column=1, sticky=tk.N)
        tk.Button(self._window, text='select xml path ', command=self.select_draw_xml_path, font=('Atril', 8)).grid(row=12, column=2, sticky=tk.NW)
        tk.Button(self._window, text='Draw', command=self.run_draw, font=('Atril', 12)).grid(row=11, column=1)

        self._window.mainloop()

    def select_image_dir(self):
        _dir = askdirectory()
        print(_dir)
        if os.path.isdir(_dir) and os.path.exists(_dir):
            self._selected_image_dir.set(_dir)
        else:
            tk.messagebox.askokcancel(title="warning", message="文件夹不存在")

    def select_xml_dir(self):
        _dir = askdirectory()
        print(_dir)
        if os.path.isdir(_dir) and os.path.exists(_dir):
            self._selected_xml_dir.set(_dir)
        else:
            tk.messagebox.askokcancel(title="warning", message="文件夹不存在")


    def select_save_image_dir(self):
        _dir = askdirectory()
        print(_dir)
        if os.path.isdir(_dir) and os.path.exists(_dir):
            self._save_image_dir.set(_dir)
        else:
            tk.messagebox.askokcancel(title="warning", message="文件夹不存在")

    def select_save_xml_dir(self):
        _dir = askdirectory()
        print(_dir)
        if os.path.isdir(_dir) and os.path.exists(_dir):
            self._save_xml_dir.set(_dir)
        else:
            tk.messagebox.askokcancel(title="warning", message="文件夹不存在")
    
    def select_draw_image_path(self):
        _image_path = askopenfilename()
        if os.path.exists(_image_path):
            self._draw_image_path_var.set(_image_path)
        else:
            tk.messagebox.askokcancel(title="warning", message="文件不存在")        

    def select_draw_xml_path(self):
        _xml_path = askopenfilename()
        if os.path.exists(_xml_path):
            self._draw_xml_path_var.set(_xml_path)
        else:
            tk.messagebox.askokcancel(title="warning", message="文件不存在")     

    def run_data_augment(self):
        ret = main(self._selected_image_dir.get(), self._selected_xml_dir.get(), self._save_image_dir.get(), self._save_xml_dir.get())
        if ret == True:
            self._finished.set("Data augmentation finished! ")

    def run_draw(self):
        image = draw_test(self._draw_image_path_var.get(), self._draw_xml_path_var.get())
        image_tkshow = ImageTk.PhotoImage(image.resize((280, 240)))
        self._image_tklabel.config(image=image_tkshow)
        self._image_tklabel.image = image_tkshow

if __name__ == "__main__":
    App()
