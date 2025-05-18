#!/usr/bin/python3
# -*- coding: utf-8 -*-


from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys,os
import numpy as np
import label_data
import cv2

# class Communicate(QObject):

font_info = QFont('Arial', 14)


from PyQt5.QtGui import QImage
import numpy as np
import cv2




def make_brush_cursor(size: int, color: QColor = Qt.black, mode: str = "brush", alpha: int = 50) -> QCursor:
    # Ensure the cursor has enough space
    pixmap_size = size + 4  # some margin
    pixmap = QPixmap(pixmap_size, pixmap_size)
    pixmap.fill(Qt.transparent)

    
    painter = QPainter(pixmap)
    
    if mode == "brush":
        # Semi-transparent fill color
        fill_color = QColor(color)
        fill_color.setAlpha(alpha)
        
        painter.setPen(color)
        painter.setBrush(QBrush(fill_color))
        
        painter.drawEllipse(2, 2, size, size)
        

    elif mode == "erase":
        painter.setPen(Qt.red)
        painter.setBrush(Qt.transparent)
        painter.drawEllipse(2, 2, size, size)
    painter.end()
    # Center the hotspot at the center of the circle
    hotspot = QPoint(pixmap_size // 2, pixmap_size // 2)
    return QCursor(pixmap, hotspot.x(), hotspot.y())


def make_point_cursor(radius=4, color=Qt.red) -> QCursor:
    size = radius * 4
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setPen(QPen(color, 1))
    painter.setBrush(QColor(color))

    center = QPoint(size // 2, size // 2)
    painter.drawEllipse(center, radius, radius)

    # cross
    painter.setPen(QPen(color, 1))
    painter.drawLine(center.x(), 0, center.x(), size)
    painter.drawLine(0, center.y(), size, center.y())

    painter.end()

    return QCursor(pixmap, center.x(), center.y())


class LabelPanel(QWidget):
    """The annotation or label panel
    
    Use QPixmap as the canvas
    """
    brush_pixel_sizes = [6, 10, 20, 40]
    
    def __init__(self,  data = None):
        super().__init__()

        self.initUI(data = data)


    def initUI(self, data ):
        """init the class

        Args:
            data (_type_): The relabelData
        """

        self.state_moving = False

        self.state_drawing_contour = False

        self.state_place_pt = False

        self.state_place_outline = False

        if data is not None:
            self.data = data

        self.act_origin_size = QAction("Original Scale", self, shortcut="Ctrl+o", triggered=self.origin, enabled = False)
        self.act_zoom_in = QAction("Zoom &In", self, shortcut="Ctrl+R", triggered=self.zoom_in, enabled = False)
        self.act_zoom_out = QAction("Zoom &Out", self, shortcut="Ctrl+F", triggered=self.zoom_out, enabled = False)

        # self.addAction(act_zoom_in)
        # self.addAction(act_zoom_out)
        # self.addAction(act_origin_size)

        self.pixmap = None
        self.scale = 1
        self.offset = QPointF(0, 0)


        self.canvas = QPixmap(self.rect().width() , self.rect().height())
        color = QColor(0, 0, 0, 0)
        self.canvas.fill(color)

        self.pt_cursor = make_point_cursor(radius=4, color=Qt.red)

        ## The numpy format of the segmentation
        # self.segmentation


        self.setMouseTracking(True)
        self.setGeometry(300, 300, 800, 600)
        self.setWindowTitle('label panel')
        # self.show()

        # Contour value, colour and category

        self.contour_colour = None
        self.contour_name = None


        self.has_no_hidden = False
        
        self.last_pos = None

    def clear_panel(self):
        self.canvas = QPixmap(self.rect().width() , self.rect().height())
        color = QColor(0, 0, 0, 0)
        self.canvas.fill(color)

    def init_for_testing(self):
        # Used for testing the widget itself.

        file_name = 'data/data_file.json'
        work_dir = './data/'

        self.data = label_data.Data(file_name,work_dir)

        # self.state_browse = True

        self.update()



    def get_brush_idx(self):
        if self.parent():
            parent = self.parent().window()

            if parent.act_brush_0.isChecked():
                return 0
            elif parent.act_brush_1.isChecked():
                return 1
            elif parent.act_brush_2.isChecked():
                return 2
            elif parent.act_brush_3.isChecked():
                return 3
            elif parent.act_brush_custom.isChecked():
                return -1
            else:
                return 0
        else:
            return 0
        
    def get_brush_width_old(self):
        if self.parent():
            parent = self.parent().window()

            if parent.act_brush_0.isChecked():               
                return 5
            elif parent.act_brush_1.isChecked():
                return 10
            elif parent.act_brush_2.isChecked():
                return 20
            elif parent.act_brush_3.isChecked():
                return 50
            else:
                return 5

        else:
            return 5

    def get_brush_width(self):
        return self.get_current_brush_size()

    def get_current_brush_pixel_size(self):
        brush_id = self.get_brush_idx()
        if brush_id == -1:
            # Custom size
            if self.parent():
                parent = self.parent().window()
                return parent.spin_brush_size.value()
            else:
                return 10
        else:
            # Predefined sizes
            return self.brush_pixel_sizes[brush_id]
    
    def get_current_brush_size(self):
        """Get the current brush size

        Returns:
            int: The current brush size
        """
        brush_id = self.get_brush_idx()
        if brush_id == -1:
            # Custom size
            if self.parent():
                parent = self.parent().window()
                pixel_size =  parent.spin_brush_size.value()
            else:
                pixel_size =  10
        else:
            # Predefined sizes
            pixel_size = self.brush_pixel_sizes[brush_id]
    
    
        scale = self.data.scale

        brush_size = int(pixel_size * scale)
        if brush_size<1:
            brush_size = 1
        
        return brush_size


    def get_brush_cate(self):
        if self.parent():
            parent = self.parent().window()
            if parent.act_brush_object.isChecked():
                return 1
            else:
                return 0
        else:
            return 1


    def get_annotation_mode(self):
        """
        Get the annotation mode (e.g. idle, Point, and segmentation)

        """

        if self.parent():
            parent = self.parent().window()

            self.state_place_outline =  parent.act_outline_mode.isChecked()
            self.state_place_pt = parent.act_point_mode.isChecked()

    def map_display_to_image_coords(self, pos):
        """
        Convert mouse position on the QWidget to image coordinates.
        
        Args:
            pos (QPoint): position from mouseEvent.pos()

        Returns:
            (x_img, y_img): int coordinates on original image
        """
        scale = self.data.scale
        x_display = pos.x()
        y_display = pos.y()

        x_img = int(x_display / scale)
        y_img = int(y_display / scale)

        h, w = self.data.current_mask.shape[:2]
        # Clip to bounds
        x_img = min(max(x_img, 0), w - 1)
        y_img = min(max(y_img, 0), h - 1)

        return x_img, y_img



    def apply_brush(self, pos, value ):
        """Apply a brush to the image at the given coordinates.
        """
        image_x, image_y = self.map_display_to_image_coords(pos)
        # Create a mask for the brush
        cv2.circle(self.data.current_mask, (image_x, image_y), self.get_current_brush_pixel_size()//2, color=int(value), thickness=-1)

    def apply_line_on_mask(self, p1, p2, value ):
        """Draw a line on the mask between two points.
        """
   
        x1, y1 = self.map_display_to_image_coords(p1)
        x2, y2  = self.map_display_to_image_coords(p2)
        
        # draw line with brush thickness
        cv2.line(self.data.current_mask, 
                 (x1, y1), (x2, y2),
                 color=int(value), 
                 thickness=self.get_current_brush_pixel_size())

    def segmentation_to_pixmap(self,
                               segmentation: np.ndarray, canvas_w: int, canvas_h: int) -> QPixmap:
        """
        Convert a segmentation mask to a colored QPixmap with transparency, resized to canvas size.
        """
        if self.parent():
            parent = self.parent().window()
            
            # Step 1: get the idx to visualize
            segs_name_id_map = self.data.get_current_image_seg_map()
            to_show_idx =[]
            for i in range(parent.widget_segment_list.count()):
                item = parent.widget_segment_list.item(i)
                if item.checkState() == Qt.Checked:
                    to_show_idx.append(segs_name_id_map[item.text()])


            
            # Step 2: Resize to canvas size
            seg_resized = cv2.resize(segmentation, (canvas_w, canvas_h), interpolation=cv2.INTER_NEAREST)

            # Step 3: Define color map
            
            color_map = parent.seg_colours_np

            # Step 4: Create RGBA image
            rgba = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
            for label in np.unique(seg_resized):
                if label == 0 or label not in to_show_idx:
                    continue
                rgba[seg_resized == label] = color_map[label % len(color_map)]

            # Step 5: Convert to QImage and QPixmap
            qimage = QImage(rgba.data, canvas_w, canvas_h, 4 * canvas_w, QImage.Format_RGBA8888)
            return QPixmap.fromImage(qimage.copy())
        

    def update_canvas(self):
        print("when mouse release, updat the canvas using the self.current mask")
        pixmap = self.segmentation_to_pixmap(self.data.current_mask, 
                                             self.canvas.width(), self.canvas.height())
        # Then paint it:
        painter = QPainter(self.canvas)
        painter.setCompositionMode(QPainter.CompositionMode_Source)
        painter.drawPixmap(0, 0, pixmap)
        painter.end()


        
    def mousePressEvent(self, e):
        """_summary_

        Args:
            e (_type_): _description_
        """
        # pos = self.coords_tranform_widget_to_image(e.pos())

        # update the annotation mode
        self.get_annotation_mode()

        pos =e.pos()

        if e.button() == Qt.LeftButton:
            mouse_in_pt = False
            # if it is on the a current point
            if self.state_place_pt :
                points = self.data.get_current_scaled_points()
                if points:
                    self.data.get_current_image().set_current_highlight_key(None)
                    for pt in points:
                        if not pt.absence:
                            if pt.rect.contains(pos):
                                self.open_state_moving()
                                self.data.get_current_image().set_current_pt_key(pt.pt_name)
                                self.data.get_current_image().set_current_highlight_key(pt.pt_name)

                                # Used to undo
                                self.data.cache_for_dragging(begin =True)
                                mouse_in_pt = True
                # Add point and Input point name
                if mouse_in_pt is False:
                    self.add_pt(pos)


            # Draw contour
            if self.state_place_outline and (self.contour_colour is not None) and (self.contour_name is not None):
                # self.draw_on_mask(pos.x(), pos.y(), brush_size=self.get_brush_width(), seg = self.get_brush_cate())
                
                self.draw_on_mask_v2(pos, pos, brush_size=self.get_brush_width(), seg = self.get_brush_cate())
                if self.get_brush_cate() ==1:
                    segs_name_id_map = self.data.get_current_image_seg_map()
                    idx = segs_name_id_map[self.contour_name]
                    self.apply_brush(pos, idx)
                else:
                    self.apply_brush(pos, 0)
                
                self.state_drawing_contour = True
                
                self.last_pos = pos
                
                # p.drawPixmap(0,0, self.pixmap)
                # p.drawPixmap(0,0, self.canvas)
                # # self.data.set_current_segment_of_current_img(pos.x(), pos.y(), self.get_brush_width(), seg = self.get_brush_cate())


            # self.data.set_current_outline_of_current_img(pos.x(),pos.y(), 10)

            # outline contour part.
            #mouse_in_contour = False
            # if it is on the a current point
            # outlines = self.data.get_current_image_scaled_outlines()


        self.update_in_parent(pos, True)
        self.update()


    def mouseReleaseEvent(self, e):

        # update the annotation mode
        self.get_annotation_mode()

        pos = e.pos() #self.coords_tranform_widget_to_image(e.pos())
        if e.button() == Qt.LeftButton:
            # if it is on the a current point
            if self.state_moving:
                self.data.cache_for_dragging(begin =False)

            self.close_state_moving()

            # For segmentation
            if self.state_place_outline and (self.contour_colour is not None) and (self.contour_name is not None):
                self.state_drawing_contour = False

                # # Update contour in relabel data
                # self.data.trans_current_segment_to_contour(self.canvas, self.contour_name)
                # self.data.trans_current_segment_to_contour_cv(self.canvas, self.contour_name, self.contour_colour)
                
                
                # self.data.trans_current_segment_to_contour_with_map(self.canvas, self.contour_name)

                self.update_canvas()
                self.data.save_as_countours()

            self.last_pos = None

        self.update_in_parent(pos)
        self.update()



    def mouseMoveEvent(self, e):
        # update the annotation mode
        self.get_annotation_mode()


        pos = e.pos() #self.coords_tranform_widget_to_image(e.pos())


        if self.state_moving:
            # Set the point with dragging on
            x = max(min(self.pixmap.rect().width(), pos.x()),0)
            y = max(min(self.pixmap.rect().height(), pos.y()), 0)

            #update the position.
            self.data.set_current_pt_of_current_img(x = x, y= y, scaled_coords = True)


        # if self.state_place_outline and self.state_drawing_contour:
        #     # Drawing contour
        #     self.draw_on_mask(pos.x(), pos.y() ,brush_size=self.get_brush_width(), seg = self.get_brush_cate())
        #     # self.data.set_current_segment_of_current_img(pos.x(), pos.y(), self.get_brush_width(), seg = self.get_brush_cate())

        if self.state_place_outline and e.buttons() == Qt.LeftButton \
            and (self.contour_colour is not None) and (self.contour_name is not None):
            # Drawing contour
            # self.draw_on_mask(pos.x(), pos.y() ,brush_size=self.get_brush_width(), seg = self.get_brush_cate())
            self.draw_on_mask_v2(self.last_pos, pos ,brush_size=self.get_brush_width(), seg = self.get_brush_cate())
            
            if self.get_brush_cate() ==1:
                segs_name_id_map = self.data.get_current_image_seg_map()
                idx = segs_name_id_map[self.contour_name]
                
                self.apply_line_on_mask(self.last_pos, pos, idx)
            else:
                self.apply_line_on_mask(self.last_pos, pos, 0)
            

            self.last_pos = pos
            # self.data.set_current_segment_of_current_img(pos.x(), pos.y(), self.get_brush_width(), seg = self.get_brush_cate())


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



    def paintEvent(self, e):
        """Paint the canvas

        Args:
            e (QPaintEvent): event
        """


        painter = QPainter(self)

        # Draw image part
        if self.data.has_images() and self.has_no_hidden:
            self.draw_image(painter)

            # Draw annotations:
            pen = QPen(Qt.red, 2)
            brush = QBrush(QColor(0, 255, 255, 120))
            painter.setPen(pen)
            painter.setBrush(brush)


           
            #### Drawing points ####

            # From the scale point , draw points
            if self.data.get_current_scaled_points():
                for pt in self.data.get_current_scaled_points():
                    if not pt.absence:
                        bbox = pt.rect
                        
                        self.draw_point(painter ,bbox , pt.pt_name)

                ##Setting for drawings Highlight:
                pen = QPen(Qt.red, 4)
                brush = QBrush(QColor(0, 255, 255, 200))
                painter.setPen(pen)
                painter.setBrush(brush)

                highlight_bbox = self.data.get_current_image().get_current_highlight_bbox(scale = self.data.scale)
                if highlight_bbox:
                    painter.drawRect(highlight_bbox)


            # Draw canvas every time
            img_size = self.data.get_current_scaled_pixmap().size()
            #
            self.canvas = self.canvas.scaled(img_size , aspectRatioMode    = Qt.IgnoreAspectRatio)
            painter.drawPixmap(0,0, self.canvas)

        else:
            painter.eraseRect(self.rect())

    
    def draw_image(self,painter):

        self.pixmap = self.data.get_current_scaled_pixmap()

        size = self.size()
        img_size = self.pixmap.size()

        if size != img_size:
            self.resize(img_size)

        painter.drawPixmap(0,0, self.pixmap)

        # If it is outline mode.
        # Draw the masked



    def draw_point(self,painter, bbox ,name):
        """Draw points on the canvas
        Args:
            painter (QPainter): QPainter
            bbox (_type_): The bounding box (scaled) of th current point
        """
        
        
        painter.setFont(font_info)
        font_size = QFontMetrics(font_info)
        
        # painter.drawEllipse(bbox.center(), TEMP_SHAPE_LENGTH/2 * 1/self.scale, TEMP_SHAPE_LENGTH/2 * 1/self.scale)
        painter.drawEllipse(bbox.center(), bbox.width()//2, bbox.height()//2)
       
        painter.drawText(bbox.center().x() - font_size.width(name)//2,
                    bbox.center().y() - font_size.height()//2,
                    name)


    def draw_polygons(self,painter, contours , colour = QColor(0, 255, 255, 200)):
        """Draw the polgyon from inside to outside (Hierarchy high to low).

        :param painter: Qpainter
        :param contours: Contours {"id": {'coords':[] , "level": , "child" :}}
        :param hierarchy:
        :param colour:
        :return:
        """

        # contours_sorted = sorted(contours.items(), key=lambda kv: kv[1]["level"] , reverse=True)
        #
        # for key, contour in contours_sorted:
        #     child = contour["child"]
        #
        #     if contour["level"] % 2 ==0:
        #         draw_colour = colour
        #     else:
        #         draw_colour = QColor(0, 255, 255, 0)
        #
        #     if child is None:
        #         self.draw_polygon(painter, contour["coords"], colour = draw_colour)
        #     else:
        #         coords_child = [contours[str(c)]["coords"] for c in child]
        #         self.draw_polygon(painter, contour["coords"] , contours_subtract = coords_child ,
        #                       colour = draw_colour)
        painter = QPainter(self.canvas)
        painter.setCompositionMode(QPainter.CompositionMode_Source)
        contours_sorted = sorted(contours.items(), key=lambda kv: kv[1]["level"])

        for key, contour in contours_sorted:
            child = contour["child"]

            if contour["level"] % 2 ==0:
                draw_colour = colour
            else:
                draw_colour = QColor(0, 0, 0, 0)


            self.draw_polygon(painter, contour["coords"], colour = draw_colour)


        # level = hierarchy["level"]
        # childs = hierarchy["child"]
        #
        # childs_levels_sorted =  sorted(zip(level,childs),  reverse=True)
        # contours_sorted = sorted(contours, key = lambda v : level,  reverse=True)
        #
        # print(childs_levels_sorted, contours)
        #
        # for contour, child_level in zip(contours_sorted,childs_levels_sorted):
        #
        #     child = child_level[1]
        #     level = child_level[0]
        #     if level % 2 ==0:
        #         draw_colour = colour
        #     else:
        #         draw_colour = QColor(0, 255, 255, 0)
        #
        #     if child ==-1:
        #         self.draw_polygon(painter, contour, colour = draw_colour)
        #     else:
        #         self.draw_polygon(painter, contour,contours[child],  colour = draw_colour)



    def draw_polygon(self,painter, contour , contours_subtract = None , colour = QColor(0, 255, 255, 200)):
        """
        Draw polygon with point and line on canvas

        :param contour: [{'x':<x> , 'y':<y>}]
        :return:
        """


        pts = [QPoint(pt['x'] , pt['y']) for pt in contour]

        # Set the colour of drawing
        # Q Pen equals to 0, so seg won't increase after every drawing.
        pen = QPen(colour, 0)
        brush = QBrush(colour)
        painter.setPen(pen)
        painter.setBrush(brush)

        poly = QPolygon(pts)
        if contours_subtract is not None:
            for contour_sub in contours_subtract:
                pts_sub = [QPoint(pt['x'] , pt['y']) for pt in contour_sub]
                poly = poly.subtracted(QPolygon(pts_sub))
        painter.drawPolygon(poly)
        # painter.drawPolyline(pts,len(pts))

    def draw_seg_cv(self,img_cv_draw,contour_cv,color):
        """
        Draw opencv image using contours
        Convert img to QPixmap

        :param contour_cv:
        :return:
        """
        resize_height = self.data.get_current_scaled_pixmap().height()
        resize_width = self.data.get_current_scaled_pixmap().width()

        if contour_cv is not None:
            contour_cv = [np.array(contour, dtype='int32') for contour in contour_cv]
            cv_colour = (color.red() , color.green() , color.blue() ,color.alpha())
            print("cv colour in LabelPanel.draw_seg_cv" , cv_colour)
            cv2.fillPoly(img_cv_draw, contour_cv, cv_colour)

            img_cv_draw = cv2.resize(img_cv_draw,(resize_width , resize_height))
            img_cv_draw[img_cv_draw[...,3] >0 , : ] = cv_colour





    def draw_on_mask(self ,x,y,brush_size,  seg):
        p_mask = QPainter(self.canvas)
        p_mask.setCompositionMode(QPainter.CompositionMode_Source)

        if seg == 1:
            # colour = QColor(0, 255, 255, 100)
            # Draw the segmentation based on contour colour
            colour = self.contour_colour
        else:
            colour = QColor(0, 0, 0, 0)

        # print("drawing colour:", colour.value())
        pen = QPen(colour, 4)
        brush = QBrush(colour)
        p_mask.setPen(pen)
        p_mask.setBrush(brush)

        p_mask.drawEllipse(x , y, brush_size, brush_size)

        # pen = QPen(colour, brush_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        # p_mask.setPen(pen)
        # p_mask.drawLine(x, y)

    def draw_on_mask_v2(self ,start_pos , end_pos,brush_size,  seg):
        p_mask = QPainter(self.canvas)
        p_mask.setCompositionMode(QPainter.CompositionMode_Source)

        if seg == 1:
            # colour = QColor(0, 255, 255, 100)
            # Draw the segmentation based on contour colour
            colour = self.contour_colour
        else:
            # Eraser
            colour = QColor(0, 0, 0, 0)
        if self.contour_colour is None:
            return
        
        # print("drawing colour:", colour.value())

        pen = QPen(colour, brush_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        p_mask.setPen(pen)
        p_mask.drawLine(start_pos, end_pos)



    def draw_init_mask(self, items ,  colors):
        """
        called in update_segment_drawing()

        :param color: dict of seg idx: segment contours
        :return:
        """

        segments_cv = self.data.get_current_image_segments_cv()


        height = self.data.get_current_origin_pixmap().height()
        width = self.data.get_current_origin_pixmap().width()



        img_cv_draw = np.zeros((height, width ,4)).astype('uint8')
        img_cv_draw = cv2.cvtColor(img_cv_draw,cv2.COLOR_BGRA2RGBA)

        # Draw the segmentations using segmentation cv
        if segments_cv:
            if self.parent():
                for item in items:
                    key = item.text()
                    contour_cv = segments_cv[key]['contours']
                    self.draw_seg_cv(img_cv_draw,contour_cv , colors[key])
                # for key, color in colors.items():
                #     contour_cv = segments_cv[key]['contours']
                #     self.draw_seg_cv(img_cv_draw,contour_cv , color)

        image_cv_draw = QImage(img_cv_draw, img_cv_draw.shape[1],\
        img_cv_draw.shape[0], img_cv_draw.shape[1] * 4,QImage.Format_RGBA8888)
        self.canvas  = QPixmap(image_cv_draw)

    def reset_mask(self, reset_data_current_mask = False):
        """
        Reset the canvas mask
        Called in file_list_current_item_changed() or update_segment_drawing()
        :return:
        """
        img_size = self.data.get_current_scaled_pixmap().size()
        self.canvas = self.canvas.scaled(img_size , aspectRatioMode    = Qt.IgnoreAspectRatio)
        self.canvas.fill(QColor(0, 0, 0, 0))
        if reset_data_current_mask:
            # Reset the background mask as well
            self.data.current_mask = None

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
        """zoom in the image
        """
        self.data.set_scale(1.25)

        
        
        self.update_brush_cursor()
        self.update()

    def zoom_out(self):
        """zoom out the image
        """       
        self.data.set_scale(0.8)
        self.update_brush_cursor()
        self.update()

    def origin(self):
        """set to the original scale
        """
        self.data.reset_scale()
        self.update_brush_cursor()
        self.update()




    def update_brush_cursor(self):
        self.get_annotation_mode()
        if self.state_place_outline:
            try:
                brush_size = self.get_current_brush_size()
                if self.get_brush_cate() == 1:
                    if self.contour_colour is not None:
                        cursor = make_brush_cursor(size = brush_size, color = self.contour_colour)
                    else:
                        cursor = make_brush_cursor(size = brush_size)
                else:
                    cursor = make_brush_cursor(size = brush_size,  mode = "erase")
                self.setCursor(cursor)
            except Exception as e:
                print("Error in update_brush_cursor: ", e)
                self.unsetCursor()

    def open_state_moving(self):
        self.state_moving = True

    def close_state_moving(self):
        self.state_moving = False


    def add_pt(self, pos):
        """add a point to self.data

        Args:
            pos (_type_): position of the point
        """
        if self.parent() and \
                self.parent().window().act_quick_label_mode.isChecked():
            cur_pt_num = len(self.data.get_current_image_points())
            quick_pt_num = len(self.parent().window().current_quick_points)
            if cur_pt_num< quick_pt_num:
                name = self.parent().window().current_quick_points[cur_pt_num]
            else:
                QMessageBox.about(self, "Warning", "You are adding more points than in the quick label mode.")
                name = self.get_annotation_name(mode='pt')
        else:
            name = self.get_annotation_name(mode='pt')
        print(self.data.get_current_image())
        if name:
            if name is None or name =='':
                QMessageBox.about(self, "Failed", "Fail to add the label\nname is empty.")
            elif self.data.add_pt_for_current_img(name, pos.x(), pos.y()):
                self.data.get_current_image().set_current_pt_key(name)
            else:
                QMessageBox.about(self, "Failed", "Fail to add the label\nname is duplicate.")

            if self.parent():
            # Add point successful and add points
                parent = self.parent().window()
                parent.list_point_name()




    def get_annotation_name(self , mode):
        """
        Dialog for inputting point name

        :return:
        """

        # List of name choices with[None, all known names of this dataset.]
        if mode == 'pt':
            items = [None]+sorted(self.data.pt_names)
        if mode == 'seg':
            items = [None]+sorted(self.data.seg_names)

        item, ok = QInputDialog().getItem(self, 'Enter the name for this label',
         "Name:", items, current = 0, editable  = True)

        if ok:
            return item
        else:
            return False


if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = LabelPanel()
    ex.init_for_testing()
    sys.exit(app.exec_())
