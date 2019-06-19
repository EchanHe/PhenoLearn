#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
ZetCode PyQt5 tutorial

In this example we draw 6 lines using
different pen styles.

Author: Jan Bodnar
Website: zetcode.com
Last edited: August 2017
"""

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys,os

import mainWin

TEMP_SHAPE_LENGTH =10
TEMP_DOUBLE_SHAPE_LENGTH = TEMP_SHAPE_LENGTH*2


class RectwithInfo(QRect ):

    def __init__(self, cx ,cy, width,height , name='' , info_dict = {}):

        x = cx - width//2
        y = cy - height//2
        super().__init__(x, y, width, height)

        self.name = name
        self.info_dict = info_dict


class LabelPanel(QWidget):

    def __init__(self,  data = None):
        super().__init__()

        self.initUI(data = data)


    def initUI(self, data ):

        self.state_moving = False
        self.state_highlight = False

        if data is not None:
            self.data = data
        # self.data.get_current_image().set_current_pt(x=7,y=8)

        act_origin_size = QAction("origin", self, shortcut="Ctrl+o",
                                      triggered=self.origin)

        act_zoom_in = QAction("Zoom &In", self, shortcut="Ctrl+R",
                                      triggered=self.zoom_in)
        act_zoom_out = QAction("Zoom &Out", self, shortcut="Ctrl+F",
                                      triggered=self.zoom_out)

        self.addAction(act_zoom_in)
        self.addAction(act_zoom_out)
        self.addAction(act_origin_size)

        self.pixmap = None
        self.scale = 1
        self.offset = QPointF(0, 0)


        self.highlight_shapes=[]

        # scroll = QScrollArea()
        # scroll.setWidget(self)
        # scroll.setWidgetResizable(True)

        self.setMouseTracking(True)
        self.setGeometry(300, 300, 800, 600)
        self.setWindowTitle('Pen styles')
        self.show()

    def init_for_testing(self):

        file_name = 'data/data_file.json'
        work_dir = './data/'

        self.data = mainWin.Data(file_name,work_dir)

        self.update()


    def mousePressEvent(self, e):
        pos = self.coords_tranfrom_widget_to_image(e.pos())

        self.data.get_current_image().set_current_highlight_id(None)

        if e.button() == Qt.LeftButton:
            # if it is on the a currrent point
            for idx, pt, in enumerate(self.data.get_current_scaled_points()):
                if pt.rect.contains(pos):
                    self.open_state_moving()
                    self.state_highlight = True
                    self.data.get_current_image().set_current_pt_id(idx)
                    self.data.get_current_image().set_current_highlight_id(idx)


            # if self.parent():
            #     self.parent().window().list_properties()


        #transform coord to image-wise coord

        # self.points_list.append(e.pos())
        #
        # bbox = QRect(0 , 0,TEMP_SHAPE_LENGTH ,TEMP_SHAPE_LENGTH)
        # bbox.moveCenter(e)
        # self.bbox_list.append(bbox)

        # select_
        self.update_in_parent(pos)
        self.update()


    def mouseReleaseEvent(self, e):
        pos = self.coords_tranfrom_widget_to_image(e.pos())
        if e.button() == Qt.LeftButton:
            # if it is on the a currrent point
            self.close_state_moving()
        self.update_in_parent(pos)
        self.update()

    def mouseMoveEvent(self, e):
        pos = self.coords_tranfrom_widget_to_image(e.pos())


        # if self.parent():
        #     parent = self.parent().window()
        #
        #     parent.label_xy_status.setText('X: %d; Y: %d' % (pos.x(), pos.y()))



        is_image_contain = self.pixmap.rect().contains(pos)

        # print("image size", self.pixmap.rect().size())
        # print("size:", self.size())
        # print("origin: {}.   trans_pos:{}".format(e.pos() , pos))
        if self.state_moving:
            x = max(min(self.pixmap.rect().width(), pos.x()),0)
            y = max(min(self.pixmap.rect().height(), pos.y()), 0)

            # Unscaled the position and update the result.
            self.data.get_current_image().set_current_pt(x = x , y= y , scale = self.data.scale)


                #
                # if self.parent():
                #     self.parent().window().list_properties()


        # self.highlight_shapes = []
        # for bbox in self.bbox_list:
        #     if bbox.contains(pos):
        #         print(pos)
        #         self.highlight_shapes.append(bbox)
        self.update_in_parent(pos)
        self.update()

    def update_in_parent(self, pos):
        if self.parent():
            parent = self.parent().window()

            scale = self.data.scale
            parent.label_xy_status.setText('X: %d; Y: %d' % (pos.x()//scale, pos.y()//scale))
            parent.statusBar().showMessage("{}%".format(round(self.data.scale,2)*100))
            parent.list_properties()


    def paintEvent(self, e):

        painter = QPainter(self)

        # Draw image part
        if self.data is not None:
            # self.pixmap = QPixmap(os.path.join(self.data.work_dir,self.data.get_current_image_name()))
            self.draw_image(painter)
            # if self.pixmap:
            #     # painter.scale(self.scale, self.scale)
            #     # self.offset = self.translate_to_center()
            #     # painter.translate(self.translate_to_center())
            #     self.draw_image(painter)
            #     # painter.scale(1/self.scale, 1/self.scale)
            # # print(pixmap.width(),pixmap.height())
            #

            # Draw annotations:
            pen = QPen(Qt.red, 2)
            brush = QBrush(QColor(0, 255, 255, 120))
            painter.setPen(pen)
            painter.setBrush(brush)

            # length = max(self.size().width() , self.size().height())
            # proportion =0.05
            # length = int(proportion * length)
            # From the scale point , draw points
            for pt in self.data.get_current_scaled_points():
                bbox = pt.rect
                print("rect size",bbox.size())
                # QRect(pt.rect.topLeft(), QSize(length,length))
                self.draw_point(painter ,bbox)


            #Setting for drawings Highlight:
            pen = QPen(Qt.red, 4)
            brush = QBrush(QColor(0, 255, 255, 200))
            painter.setPen(pen)
            painter.setBrush(brush)

            highlight_bbox = self.data.get_current_image().get_current_highlight_bbox(scale = self.data.scale)
            if highlight_bbox:
                painter.drawRect(highlight_bbox)


            # if self.state_highlight:
            #     selected_bbox = self.data.get_current_image().get_current_pt().rect
            #     painter.drawRect(selected_bbox)


    def draw_image(self,painter):
        # painter.drawPixmap(self.rect(), self.pixmap)
        # self.pixmap.scaled(self.pixmap.size()*self.scale)
        # print(self.pixmap.size()*self.scale)

        # Check the


        #Check the size of pixmap first:
        # limit_size = QSize(1300,700)
        # self.pixmap = self.data.current_pixmap

        # Scale the size first from pixmap:
        # self.pixmap = self.pixmap.scaled(self.scale *  self.pixmap.size() , Qt.KeepAspectRatio)
        #
        # if self.pixmap.width()>limit_size.width() or self.pixmap.height()>limit_size.height():
        #
        #     self.pixmap  = self.pixmap.scaled(limit_size, Qt.KeepAspectRatio )
        #     print("resize map" , self.pixmap.size())

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



    def zoom_in(self):
        self.data.set_scale(1.25)

        self.update()
        # new_size = self.scale * self.pixmap.size()
        # self.resize(new_size)
        # self.update()

    def zoom_out(self):
        self.data.set_scale(0.8)
        self.update()

    def origin(self):
        self.data.reset_scale()
        self.update()

    def coords_tranfrom_widget_to_image(self,pos):
        s = self.scale
        offset = self.offset
        new_x = pos.x()/s
        new_y = pos.y()/s
        new_x -=offset.x()
        new_y -=offset.y()
        return QPoint(int(new_x) , int(new_y))

    def translate_to_center(self):
        s = self.scale
        area = super(LabelPanel, self).size()
        w, h = self.pixmap.width() * s, self.pixmap.height() * s
        aw, ah = area.width(), area.height()
        x = (aw - w) / (2 * s) if aw > w else 0
        y = (ah - h) / (2 * s) if ah > h else 0
        return QPointF(x, y)

    def toggle_state_moving(self):
        self.state_moving = not self.state_moving

    def open_state_moving(self):
        self.state_moving = True

    def close_state_moving(self):
        self.state_moving = False

    # def paintEvent(self, event):
    #     painter = QPainter(self)
    #     pixmap = QPixmap("large.jpg")
    #     painter.drawPixmap(self.rect(), pixmap)
    #     # pen = QPen(Qt.red, 3)
    #
    #
    #     pen = QPen(Qt.red, 2)
    #     brush = QBrush(QColor(0, 255, 255, 120))
    #     painter.setPen(pen)
    #     painter.setBrush(brush)
    #
    #     for i in range(self.points.count()):
    #         painter.drawEllipse(self.points.point(i), 5, 5)
    #
    #     # painter.drawLine(10, 10, self.rect().width() -10 , 10)
    #
    #
    #
    #     # painter.drawRect(40, 40, 400, 200)

    # def paintEvent(self, e):
    #
    #     qp = QPainter()
    #     qp.begin(self)
    #     pixmap = QPixmap("small.png")
    #     qp.drawPixmap(self.rect(), pixmap)
    #     self.drawLines(qp)
    #     qp.end()
    #
    #
    # def drawLines(self, qp):
    #
    #     pen = QPen(Qt.black, 2, Qt.SolidLine)
    #
    #     qp.setPen(pen)
    #     qp.drawLine(20, 40, 250, 40)
    #
    #     pen.setStyle(Qt.DashLine)
    #     qp.setPen(pen)
    #     qp.drawLine(20, 80, 250, 80)
    #
    #     pen.setStyle(Qt.DashDotLine)
    #     qp.setPen(pen)
    #     qp.drawLine(20, 120, 250, 120)
    #
    #     pen.setStyle(Qt.DotLine)
    #     qp.setPen(pen)
    #     qp.drawLine(20, 160, 250, 160)
    #
    #     pen.setStyle(Qt.DashDotDotLine)
    #     qp.setPen(pen)
    #     qp.drawLine(20, 200, 250, 200)
    #
    #     pen.setStyle(Qt.CustomDashLine)
    #     pen.setDashPattern([1, 4, 5, 4])
    #     qp.setPen(pen)
    #     qp.drawLine(20, 240, 250, 240)



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


if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = LabelPanel()
    ex.init_for_testing()
    sys.exit(app.exec_())
