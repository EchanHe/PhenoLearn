#!/usr/bin/python3
# -*- coding: utf-8 -*-


from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys,os

import mainWin

class LabelPanel(QWidget):

    def __init__(self,  data = None):
        super().__init__()

        self.initUI(data = data)


    def initUI(self, data ):

        self.state_moving = False
        self.state_highlight = False

        if data is not None:
            self.data = data

        self.act_origin_size = QAction("Original Scale", self, shortcut="Ctrl+o",
                                      triggered=self.origin)
        self.act_zoom_in = QAction("Zoom &In", self, shortcut="Ctrl+R",
                                      triggered=self.zoom_in)
        self.act_zoom_out = QAction("Zoom &Out", self, shortcut="Ctrl+F",
                                      triggered=self.zoom_out)

        # self.addAction(act_zoom_in)
        # self.addAction(act_zoom_out)
        # self.addAction(act_origin_size)

        self.pixmap = None
        self.scale = 1
        self.offset = QPointF(0, 0)

        self.setMouseTracking(True)
        self.setGeometry(300, 300, 800, 600)
        self.setWindowTitle('Pen styles')
        self.show()

    def init_for_testing(self):
        # Used for testing the widget itself.

        file_name = 'data/data_file.json'
        work_dir = './data/'

        self.data = mainWin.Data(file_name,work_dir)

        self.update()


    def mousePressEvent(self, e):
        # pos = self.coords_tranform_widget_to_image(e.pos())

        pos =e.pos()

        if e.button() == Qt.LeftButton:
            # if it is on the a currrent point
            if self.data.get_current_scaled_points():
                self.data.get_current_image().set_current_highlight_id(None)

                for idx, pt, in enumerate(self.data.get_current_scaled_points()):
                    if not pt.absence:
                        if pt.rect.contains(pos):
                            self.open_state_moving()
                            self.state_highlight = True
                            self.data.get_current_image().set_current_pt_id(idx)
                            self.data.get_current_image().set_current_highlight_id(idx)

        self.update_in_parent(pos, True)
        self.update()


    def mouseReleaseEvent(self, e):
        pos = e.pos() #self.coords_tranform_widget_to_image(e.pos())
        if e.button() == Qt.LeftButton:
            # if it is on the a currrent point
            self.close_state_moving()
        self.update_in_parent(pos)
        self.update()

    def mouseMoveEvent(self, e):
        pos = e.pos() #self.coords_tranform_widget_to_image(e.pos())


        if self.state_moving:
            x = max(min(self.pixmap.rect().width(), pos.x()),0)
            y = max(min(self.pixmap.rect().height(), pos.y()), 0)

            #update the position.
            self.data.set_current_pt_of_current_img(x = x, y= y, scaled_coords = True)


        self.update_in_parent(pos,self.state_moving)
        self.update()

    def update_in_parent(self, pos = None, moved = False):
        if self.parent():
            parent = self.parent().window()

            scale = self.data.scale
            parent.statusBar().showMessage("{}%".format(round(self.data.scale,2)*100))

            if pos is not None:
                pixel_value = parent.data.get_current_scaled_pixmap().toImage().pixel(pos.x(),pos.y())
                pixel_value = (qRed(pixel_value), qGreen(pixel_value), qBlue(pixel_value))

                parent.label_xy_status.setText('X: {}; Y: {}; RGB: {}'.format(pos.x()//scale, pos.y()//scale, pixel_value))

            if moved:
                parent.list_properties()
            if parent.data.changed:
                parent.widget_anno_file_label.setText("Annotation file: {}*".format(parent.data.file_name))


    def paintEvent(self, e):

        painter = QPainter(self)

        # Draw image part
        if self.data is not None and self.data.has_images():
            self.draw_image(painter)

            # Draw annotations:
            pen = QPen(Qt.red, 2)
            brush = QBrush(QColor(0, 255, 255, 120))
            painter.setPen(pen)
            painter.setBrush(brush)


            # From the scale point , draw points
            if self.data.get_current_scaled_points():
                for pt in self.data.get_current_scaled_points():
                    if not pt.absence:
                        bbox = pt.rect
                        self.draw_point(painter ,bbox)


                #Setting for drawings Highlight:
                pen = QPen(Qt.red, 4)
                brush = QBrush(QColor(0, 255, 255, 200))
                painter.setPen(pen)
                painter.setBrush(brush)

                highlight_bbox = self.data.get_current_image().get_current_highlight_bbox(scale = self.data.scale)
                if highlight_bbox:
                    painter.drawRect(highlight_bbox)
        else:
            painter.eraseRect(self.rect())



    def draw_image(self,painter):

        self.pixmap = self.data.get_current_scaled_pixmap()

        size = self.size()
        img_size = self.pixmap.size()

        if size != img_size:
            self.resize(img_size)

        painter.drawPixmap(0,0, self.pixmap)

    def draw_point(self,painter, bbox):
        # painter.drawEllipse(bbox.center(), TEMP_SHAPE_LENGTH/2 * 1/self.scale, TEMP_SHAPE_LENGTH/2 * 1/self.scale)
        painter.drawEllipse(bbox.center(), bbox.width()//2, bbox.height()//2)


    def wheelEvent(self,e):
        modifiers = QApplication.keyboardModifiers()
        delta = e.angleDelta()
        if self.data is not None:
            if modifiers and modifiers == Qt.ControlModifier:
                delta = e.angleDelta()
                if delta.y()>0:
                    self.zoom_in()
                elif delta.y()<0:
                    self.zoom_out()
            self.update_in_parent()

    def zoom_in(self):
        self.data.set_scale(1.25)
        self.update()

    def zoom_out(self):
        self.data.set_scale(0.8)
        self.update()

    def origin(self):
        self.data.reset_scale()
        self.update()

    def toggle_state_moving(self):
        self.state_moving = not self.state_moving

    def open_state_moving(self):
        self.state_moving = True

    def close_state_moving(self):
        self.state_moving = False

    ### Deprecated funs ###

    # # Setting the image and annotation for labelling.
    # def open_image(self , file_name = None):
    #     if file_name:
    #         self.pixmap = QPixmap(file_name)
    #         self.update()
    #
    # def set_annotations(self, pt_list):
    #
    #     if pt_list:
    #         self.selected_bbox_id = None
    #         self.bbox_list = []
    #         for pt in pt_list:
    #             rect_temp = RectwithInfo(pt.x , pt.y ,TEMP_SHAPE_LENGTH , TEMP_SHAPE_LENGTH, pt.pt_name)
    #             self.bbox_list.append(rect_temp)
    #         self.update()
    # def coords_tranform_widget_to_image(self,pos):
    #     s = self.scale
    #     offset = self.offset
    #     new_x = pos.x()/s
    #     new_y = pos.y()/s
    #     new_x -=offset.x()
    #     new_y -=offset.y()
    #     return QPoint(int(new_x) , int(new_y))

if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = LabelPanel()
    ex.init_for_testing()
    sys.exit(app.exec_())
