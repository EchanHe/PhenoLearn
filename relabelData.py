#!/usr/bin/python3
# -*- coding: utf-8 -*-

from PyQt5.QtCore import *
from PyQt5.QtGui import QPixmap, QImageReader
import os
from pathlib import Path
import json
from math import ceil

pixmap_max_size_limit = QSize(1300,700)
pixmap_min_size_limit = QSize(400,200)

rect_length_prop = 0.015
min_rect_length = 20

class Point():
    def __init__(self, pt_name, x, y, width = 10, error = False,info = None, absence = False):
        self.set_point(pt_name,x,y,width,error=error,info=info,absence=absence, save_cache=False)

    def set_point(self, pt_name = None,x= None,y=None, width = None, error = None,info = None, absence = None, save_cache = True):
        if (pt_name is not None or x is not None or y is not None or error is not None or absence is not None) and save_cache:
            self.cache = self.get_point_props_dict()
        else:
            self.cache = None

        if pt_name is not None:
            self.pt_name = pt_name
        if x is not None:
            self.x = x
        if absence is not None:
            self.absence = absence
        if y is not None:
            self.y = y
        if error is not None:
            self.error = error

        self.info = info
        if width is None:
            width = self.rect.width()

        self.set_rect(self.x,self.y, width, width)


    def set_rect(self,cx,cy,width,height):

        x = cx - width//2
        y = cy - height//2
        self.rect = QRect(x, y, width, height)

    def check_diff(self, pt_name = None,x= None,y=None, width = None, error = None,info = None, absence = None):
        cond_name = pt_name is not None and pt_name != self.pt_name
        cond_x = x is not None and x != self.x
        cond_y = y is not None and y != self.y
        cond_absence = absence is not None and absence != self.absence
        cond_error =  error is not None and error != self.error
        cond_width = width is not None and width != self.rect.width()
        cond_info = info != self.info

        return cond_name or cond_x or cond_y or cond_absence or cond_error or cond_width or cond_info


    def __mul__(self, factor):
        new_x = self.x * factor
        new_y = self.y * factor
        new_width = self.rect.width() *factor

        return Point(self.pt_name, x=new_x, y=new_y, width=new_width, error=self.error, info=self.info, absence=self.absence)

    def undo_set_point(self):
        if self.cache is not None:
            self.set_point(self.cache['pt_name'], self.cache['x'], self.cache['y'],
                           error = self.cache['error'], absence = self.cache['absence'], save_cache = False)

    def get_point_props_dict(self):
        point = {'pt_name':self.pt_name,'x':self.x,'y':self.y,'absence':self.absence,'error':self.error}
        return point


    def set_current_cache(self,x, y):
        self.cache = self.get_point_props_dict()
        self.cache['x'] = x
        self.cache['y'] = y

class Image():
    current_pt_id = None
    def __init__(self, img_name, pt_lists=None):
        self.img_name = img_name
        self.attention_flag = False
        # Points
        self.points = []
        if pt_lists:
            for pt in pt_lists:
                point = Point(pt['name'], pt['x'] , pt['y'], absence=pt.get("absence", False))
                self.points.append(point)
            self.current_pt_id = 0

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

    def set_current_pt_id(self, idx):
        """
        Change the id for point
        :param idx:
        :return:
        """
        self.current_pt_id = idx

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
    def get_curent_pt_props_dict(self):
        return self.points[self.current_pt_id].get_point_props_dict()

    # def next_point(self):
    #     if self.current_pt_id<self.pt_size-1:
    #         self.current_pt_id = (self.current_pt_id+1)
    #
    # def prev_image(self):
    #     if self.current_pt_id>0:
    #         self.current_pt_id = (self.current_pt_id-1)

class Data():
    def __init__(self, file_name = None, work_dir=None):
        """
        Init Data class (images and annotation per image)

        :param file_name: annotation file
        :param work_dir: working directory for images
        """
        self.images = []
        self.pt_names = set()

        self.work_dir = work_dir

        if file_name == None:
            self.file_name = "untitled.json"
            self.no_anno_file = True
        else:
            self.file_name = file_name
            self.no_anno_file = False
        self.changed = False
        self.init_images()


    def init_images(self):
        self.images = []
        self.pt_names = set()
        if self.work_dir is not None:
            # Get list of images
            extensions = ['.%s' % fmt.data().decode("ascii").lower() for fmt in QImageReader.supportedImageFormats()]
            work_dir_file_names = [os.path.basename(file_name) for file_name in Path(self.work_dir).glob('*')  if str(file_name).lower().endswith(tuple(extensions))]

            if not self.no_anno_file:
            # Has annotation file
                with open(self.file_name, "r") as read_file:
                    data = json.load(read_file)
                # Init images list
                for entry in data:
                    if entry['file_name'] in work_dir_file_names:
                        image = Image(entry['file_name'] , entry['points'])
                        self.images.append(image)
                        for pt in entry['points']:
                            self.pt_names.add(pt['name'])
            else:
            # without annotation file
                for name in work_dir_file_names:
                    image = Image(name)
                    self.images.append(image)


            self.current_image_id = 0
            self.img_size = len(self.images)
            if self.img_size !=0:
                self.current_pixmap = QPixmap(os.path.join(self.work_dir,self.get_current_image_name()))
                self.set_scale_fit_limit()


    def set_image_id(self, idx):
        self.current_image_id = idx
        # Reset scale and current pixmap

        self.current_pixmap = QPixmap(os.path.join(self.work_dir,self.get_current_image_name()))
        self.set_scale_fit_limit()

    def set_work_dir(self,work_dir):
        self.work_dir = work_dir
        self.init_images()

    def set_file_name(self,file_name):
        self.no_anno_file = False
        self.file_name = file_name
        self.init_images()

    def set_scale(self,scale):
        temp = self.scale * scale
        if temp <= 7* self.origin_scale and temp >= self.origin_scale/2:
            self.scale=temp

    def reset_scale(self):
        self.scale = self.origin_scale

    def set_scale_fit_limit(self):

        # Setting the maximum scale limit
        if self.current_pixmap.width()>pixmap_max_size_limit.width() or self.current_pixmap.height()>pixmap_max_size_limit.height():
            width_scale = ceil(self.current_pixmap.width()/pixmap_max_size_limit.width())
            height_scale = ceil(self.current_pixmap.height()/pixmap_max_size_limit.height())

            self.origin_scale = 1/max(width_scale,height_scale)
        elif self.current_pixmap.width()<pixmap_min_size_limit.width() or self.current_pixmap.height()<pixmap_min_size_limit.height():
            width_scale = ceil(pixmap_min_size_limit.width()/self.current_pixmap.width())
            height_scale = ceil(pixmap_min_size_limit.height()/self.current_pixmap.height())

            self.origin_scale = max(width_scale,height_scale)
        else:
            self.origin_scale = 1

        self.scale = self.origin_scale
        size = self.current_pixmap.size()

        # The width length
        length = int(rect_length_prop * max(size.width(),size.height()))

        if length<min_rect_length:
            length = min_rect_length
        self.get_current_image().set_points_width(length)

    def set_current_pt_of_current_img(self, pt_name = None,x= None,y=None, error = None, absence = None,info = None, scaled_coords= False , save_cache = True):
        if scaled_coords:
            if y is not None:
                y = int(y/self.scale)
            if x is not None:
                x = int(x/self.scale)

        if self.get_current_pt_of_current_img().check_diff(pt_name = pt_name, x= x, y=y, error = error , absence = absence,info = info):
            self.get_current_image().get_current_pt().set_point(pt_name = pt_name, x= x, y=y, error = error, absence = absence,info = info, save_cache = save_cache)

            self.changed = True

    def set_current_pt_of_current_img_dict(self, pt_prop):

        self.set_current_pt_of_current_img(pt_name = pt_prop.get('pt_name', None), x = pt_prop.get('x', None), y = pt_prop.get('y', None),
                                      absence = pt_prop.get('absence', None) , error = pt_prop.get('error', None) )



    def add_pt_for_current_img(self, pt_name, x, y, scaled_coords= True):
        """
        Add point if the name if not duplicate
        :param pt_name: pt name
        :param x: x
        :param y: y
        :param scaled_coords: if it need to scale back to real coords
        :return: If it is possible to add point
        """

        pt_names = [pt.pt_name for pt in self.get_current_image().points]
        if not pt_names:
            self.get_current_image().current_pt_id=0
        if pt_name not in pt_names:
            # if the point name is not duplicate add
            size = self.current_pixmap.size()
            length = int(rect_length_prop * max(size.width(),size.height()))

            if scaled_coords:
                y = int(y/self.scale)
                x = int(x/self.scale)

            temp = Point(pt_name, x, y, width = length, absence=False)
            self.get_current_image().points.append(temp)

            self.pt_names.add(pt_name)
            return True

        else:
            return False

    def remove_pt_for_current_img(self, idx = None):
        if idx is not None:
            self.get_current_image().points.pop(idx)
        else:
            idx = self.get_current_image().current_pt_id
            self.get_current_image().points.pop(idx)

        if self.get_current_image().current_pt_id >= len(self.get_current_image().points):
            self.get_current_image().current_pt_id = len(self.get_current_image().points)-1

    def get_current_pt_of_current_img(self):
        return self.images[self.current_image_id].get_current_pt()

    def get_current_image(self):
        if self.images:
            return self.images[self.current_image_id]
        else:
            return None

    def get_current_image_points(self):
        if self.images:
            return self.images[self.current_image_id].points
        else:
            return None

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

    def get_current_origin_pixmap(self):
        return self.current_pixmap

    def get_current_scaled_points(self):
        if self.images:
            return [pt * self.scale for pt in self.get_current_image_points()]
        else:
            return None

    def has_images(self):
        if self.images:
            return True
        else:
            return False

    def has_points_current_image(self):
        if self.images and self.images[self.current_image_id].points:
            return True
        else:
            return False

    def toggle_flag_img(self, state):
        if state == True:
            self.images_backup = self.images.copy()
            self.images = [img for img in self.images if img.attention_flag == True]
        else:
            self.images = self.images_backup
            self.images_backup = None

    def write_json(self, save_name = None):
        # Create data form to save
        image_list = []
        for image in self.images:
            entry = {'file_name': image.img_name}
            points = []
            for pt in image.points:
                pt_data = {"name": pt.pt_name, "x": pt.x, "y": pt.y, "info": pt.info,"error": pt.error, "absence": pt.absence}
                points.append(pt_data)
            entry['points'] = points
            image_list.append(entry)

        if save_name is None:
            save_name = self.file_name

        with open(save_name, 'w') as write:
            json.dump(image_list, write)

        self.changed = False



