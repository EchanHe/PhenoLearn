
from PyQt5.QtCore import *
from PyQt5.QtGui import QPixmap
import os
from pathlib import Path
import json
from math import ceil

TEMP_SHAPE_LENGTH =111

pixmap_size_limit = QSize(1300,700)


class Point():
    def __init__(self, pt_name, x, y, width = TEMP_SHAPE_LENGTH, error = False,info = None):
        self.pt_name = pt_name
        self.x = x
        self.y = y
        self.info = info
        self.error = error

        self.set_rect(self.x,self.y, width, width)


    def set_rect(self,cx,cy,width,height):


        x = cx - width//2
        y = cy - height//2
        self.rect = QRect(x, y, width, height)

    def set_point(self, pt_name = None,x= None,y=None, width = None, error = None,info = None):
        if pt_name is not None:
            self.pt_name = pt_name
        if x is not None:
            self.x = x
        if y is not None:
            self.y = y

        if error is not None:
            self.error = error
        if info is not None:
            self.info = info

        if width is None:
            width = self.rect.width()



        self.set_rect(self.x,self.y, width, width)

    def __mul__(self, factor):
        new_x = self.x * factor
        new_y = self.y * factor
        new_width = self.rect.width() *factor

        return Point(self.pt_name, x=new_x, y=new_y, width=new_width, error=self.error, info=self.info)



class Image():
    current_pt_id = None
    def __init__(self, img_name, pt_lists=None):
        self.img_name = img_name

        if pt_lists:
            self.points = []
            for pt in pt_lists:
                point = Point(pt['name'], pt['x'] , pt['y'])
                self.points.append(point)

            self.current_pt_id = 0
            self.pt_size = len(self.points)
            self.current_highlight_id = None

    def get_current_highlight_bbox(self, scale= None):
        if self.current_highlight_id is not None:
            if scale is not None:
                return (self.points[self.current_highlight_id] * scale).rect
            else:
                return self.points[self.current_highlight_id].rect
        else:
            return None
    def set_current_highlight_id(self, idx):
        self.current_highlight_id = idx

    def next_point(self):
        if self.current_pt_id<self.pt_size-1:
            self.current_pt_id = (self.current_pt_id+1)

    def prev_image(self):
        if self.current_pt_id>0:
            self.current_pt_id = (self.current_pt_id-1)


    def set_current_pt_id(self, idx):
        self.current_pt_id = idx

    def set_current_pt(self, pt_name = None,x= None,y=None, error = None,info = None, scale=1):
        if y is not None:
            y=y//scale
        if x is not None:
            x=x//scale

        self.points[self.current_pt_id].set_point(pt_name = pt_name, x= x, y=y, error = error,info = info)


    def set_current_pt_x(self, x):
        self.points[self.current_pt_id].x = x

    def set_current_pt_y(self, y):
        self.points[self.current_pt_id].y = y

    def set_points_width(self,width):
        for pt in self.points:
            pt.set_point(width=width)

    def get_current_pt_name(self):
        return self.points[self.current_pt_id].pt_name

    def get_current_pt_id(self):
        return self.current_pt_id

    def get_current_pt(self):
        return self.points[self.current_pt_id]

    def get_current_pt_x(self):
        return self.points[self.current_pt_id].x

    def get_current_pt_y(self):
        return self.points[self.current_pt_id].y



rect_length_prop = 0.015
class Data():

    def __init__(self, file_name, work_dir="."):

        self.work_dir = work_dir
        self.file_name = file_name

        self.init_images()


    def init_images(self):
        self.images = []
        work_dir_file_names = [os.path.basename(file_name) for file_name in Path(self.work_dir).glob('*')]

        with open(self.file_name, "r") as read_file:
            data = json.load(read_file)
        # Init images list
        for entry in data:
            if entry['file_name'] in work_dir_file_names:
                image = Image(entry['file_name'] , entry['points'])
                self.images.append(image)

        self.current_image_id = 0
        self.img_size = len(self.images)

        self.current_pixmap = QPixmap(os.path.join(self.work_dir,self.get_current_image_name()))
        self.set_scale_fit_limit()
        # Read and set the scale if it is too large



    # def next_image(self):
    #     if self.current_image_id<self.img_size-1:
    #         self.current_image_id = (self.current_image_id+1)
    #
    # def prev_image(self):
    #     if self.current_image_id>0:
    #         self.current_image_id = (self.current_image_id-1)

    def set_image_id(self, idx):
        self.current_image_id = idx
        # Reset scale and current pixmap

        self.current_pixmap = QPixmap(os.path.join(self.work_dir,self.get_current_image_name()))
        self.set_scale_fit_limit()

    def set_work_dir(self,work_dir):
        self.work_dir = work_dir
        self.init_images()

    def set_file_name(self,file_name):
        self.file_name = file_name
        self.init_images()

    def set_scale(self,scale):
        temp = self.scale * scale
        if temp <= 7* self.origin_scale and temp >= self.origin_scale/2:
            self.scale=temp

    def reset_scale(self):
        self.scale = self.origin_scale

    def set_scale_fit_limit(self):
        if self.current_pixmap.width()>pixmap_size_limit.width() or self.current_pixmap.height()>pixmap_size_limit.height():
            width_scale = ceil(self.current_pixmap.width()/pixmap_size_limit.width())
            height_scale = ceil(self.current_pixmap.height()/pixmap_size_limit.height())

            self.set_scale(1/max(width_scale,height_scale))
        else:
            self.scale = 1
        self.origin_scale = self.scale

        size = self.current_pixmap.size()
          # length = max(self.size().width() , self.size().height())
            # proportion =0.05
        length = int(rect_length_prop * max(size.width(),size.height()))
        print(length)
        self.get_current_image().set_points_width(length)




    def get_current_image(self):
        return self.images[self.current_image_id]

    def get_current_image_points(self):
        return self.images[self.current_image_id].points

    def get_current_image_name(self):
        return self.images[self.current_image_id].img_name

    def get_current_image_abs_path(self):
        path = os.path.join(self.work_dir , self.images[self.current_image_id].img_name)
        path = os.path.abspath(path)
        return path

    def get_image_id(self):
        return self.current_image_id

    def get_current_scaled_pixmap(self):
        return self.current_pixmap.scaled(self.scale *  self.current_pixmap.size() , Qt.KeepAspectRatio)

    def get_current_scaled_points(self):
        return [pt * self.scale for pt in self.get_current_image_points()]

    def write_json(self, save_name = None):
        # Create data form to save
        image_list = []
        for image in self.images:
            entry = {'file_name': image.img_name}
            points = []
            for pt in image.points:
                pt_data = {"name": pt.pt_name, "x": pt.x, "y": pt.y, "info": pt.info,"error": pt.error}
                points.append(pt_data)
            entry['points'] = points
            image_list.append(entry)

        if save_name is None:
            save_name = self.file_name

        with open(save_name, 'w') as write:
            json.dump(image_list, write)

