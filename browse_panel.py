from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys,os

from math import ceil, floor
import cv2
import timeit
import numpy as np

item_size = QSize(350,350)
icon_size = QSize(300,300)
thumbnail_size = QSize(300,300)
point_size = QSize(4,4)
font_info = QFont('Arial', 12)
font_size_scale= 2

class BrowsePanel(QWidget):
    """The review panel class

    Args:
        QWidget (_type_): _description_
    """

    def __init__(self,  data = None):
        super().__init__()

        self.initUI(data = data)


    def initUI(self, data):
        """Init the review panel

        """
        if data is not None:
            self.data = data
        self.layout = QVBoxLayout(self)

        # The widget set up of the browse mode view.
        # The the widget as iconmode, so it will show thumbnails of images
        self.widget_image_browser = QListWidget()
        self.widget_image_browser.setIconSize(item_size)

        self.widget_image_browser.setViewMode(QListView.IconMode)

        self.widget_image_browser.setFlow((QListView.LeftToRight))
        self.widget_image_browser.setResizeMode(QListView.Adjust)
        self.widget_image_browser.setSpacing(10)

        self.widget_image_browser.setMovement(QListView.Static)

        self.widget_image_browser.verticalScrollBar().valueChanged.connect(self.prepare_images)
        self.widget_image_browser.itemChanged.connect(self.trigger_check_state)
        self.widget_image_browser.itemDoubleClicked.connect(self.go_to_annotation)


        # self.read_img_threads = [ReadThumbnailThread([])]*3
        #
        # for thread in self.read_img_threads:
        #     thread.read_one_image.connect(self.add_image)


        self.painter = QPainter()
        ## set the size of font
        
  
        self.painter.setCompositionMode(QPainter.CompositionMode_Source)

        self.layout.addWidget(self.widget_image_browser)
        self.setLayout(self.layout)

        # self.read_thread_1 = ReadThumbnailThread()
        # self.read_thread_1.read_one_image.connect(self.add_image)
        #
        # self.read_thread_2 = ReadThumbnailThread()
        # self.read_thread_2.read_one_image.connect(self.add_image)
        self.init_list()

        self.threadpool = QThreadPool()
        self.worker = None
        print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())

        # self.setMouseTracking(True)
        # self.resize.connect(self.sizechanged)

        self.setGeometry(300, 300, 800, 600)
        self.setWindowTitle('Browse Panel')
        # self.show()

        self.setMouseTracking(True)
        self.is_icon_hide = False


    def init_list(self):
        """Init list items and their icon placeholder
        :return:
        """
        self.widget_image_browser.clear()
        if self.parent() is not None:
            parent = self.parent().window()

            for i in range(parent.widget_file_list.count()):
                item_file_list = parent.widget_file_list.item(i)
                if not item_file_list.isHidden():
                    item = QListWidgetItem()
                    item.setSizeHint(item_size)
                    item.setText(item_file_list.text())
                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                    if self.data.images[str(item.text())].attention_flag:
                        item.setCheckState(Qt.Checked)
                    else:
                        item.setCheckState(Qt.Unchecked)

                    self.widget_image_browser.addItem(item)
        # for key, image in images.items():
        #     # QT part
        #     item = QListWidgetItem()
        #     item.setSizeHint(item_size)
        #     # item.setIcon(icon)
        #     item.setText(image.img_name)
        #     item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        #     if image.attention_flag:
        #         item.setCheckState(Qt.Checked)
        #     else:
        #         item.setCheckState(Qt.Unchecked)



        self.thumbnail_list = [None] * self.data.img_size

        # self.thumbnail_list = {}



        # self.read_thread_1.work_dir = self.data.work_dir
        # self.read_thread_2.work_dir = self.data.work_dir

        self.prepare_images()


    def reset_widget(self):
        self.init_list()


    #### add image and draw all annotations #####

    def add_image(self, args):
        """Draw thumbnail images, points and segmentation
        No multi_threads version
    
        """

        pixmap = args[1]
        img_id = args[0]
        points_dict = args[2]
        segments_cv = args[3]
        img_name_id = self.data.img_id_order[img_id]
        # print(img_id, pixmap is not None , self.data.images[img_id].label_changed)
        # if not (pixmap is None and self.data.images[img_name_id].label_changed == False):
        if pixmap is not None:
            self.thumbnail_list[img_id] = pixmap.copy()
        # elif self.data.images[img_name_id].label_changed:
        else:
            pixmap = self.thumbnail_list[img_id].copy()
            self.data.images[img_name_id].label_changed = False

        
        self.painter.begin(pixmap)
        self.draw_points_with_painter(self.painter,points_dict)
        self.painter.drawPixmap(0,0,self.draw_seg_args(segments_cv))
        self.painter.end()

        pixmap = pixmap.scaled(icon_size, aspectRatioMode=Qt.KeepAspectRatio)
        icon = QIcon()
        icon.addPixmap(pixmap, QIcon.Normal, QIcon.Off)
        self.widget_image_browser.item(img_id).setIcon(icon)



    def add_image_QRunnable(self,img_list , img_id_list):
        """Draw thumbnail images, points and segmentation
        multi_threads version using QRunnable

        """
        ### Read Pixmap, original readthumbnail_thread.Run()
        for image, img_id in zip(img_list , img_id_list):
            points_dict = image.points_dict
            segments_cv = image.segments_cv

            if self.widget_image_browser.item(img_id).icon().isNull():
            # If the icon is NUll, read images

            # read images and draw annotation
            # Read images
                pixmap = QPixmap(os.path.join(self.data.work_dir, image.img_name))
                pixmap = pixmap.scaled(thumbnail_size, aspectRatioMode=Qt.KeepAspectRatio)
            else:
                pixmap = None

        #####

            img_name_id = self.data.img_id_order[img_id]
            # print(img_id, pixmap is not None , self.data.images[img_id].label_changed)
            if not (pixmap is None and self.data.images[img_name_id].label_changed == False):
                if pixmap is not None:
                    self.thumbnail_list[img_id] = pixmap.copy()
                elif self.data.images[img_name_id].label_changed:
                    pixmap = self.thumbnail_list[img_id].copy()
                    self.data.images[img_name_id].label_changed = False

                # print("browser panel: " + img_name_id)
                # print(img_id)
                # print(points_dict)
               
                self.painter.begin(pixmap)
                self.draw_points_with_painter(self.painter,points_dict)
                self.painter.drawPixmap(0,0,self.draw_seg_args(segments_cv))
                self.painter.end()
                
                pixmap = pixmap.scaled(icon_size, aspectRatioMode=Qt.KeepAspectRatio)
                icon = QIcon()
                icon.addPixmap(pixmap, QIcon.Normal, QIcon.Off)
                self.widget_image_browser.item(img_id).setIcon(icon)


    def draw_points_with_painter(self , painter,points_dict):
            """Draw keypoint based on the scale (original reso / thumbnail reso)

            Args:
                points_dict (_type_): A dictionary of points detail
            """        


            scale = max(self.data.current_pixmap.width()/thumbnail_size.width(), self.data.current_pixmap.height()/thumbnail_size.height())

            pen = QPen(Qt.red, 2)
            brush = QBrush(QColor(0, 255, 255, 120))
            painter.setPen(pen)
            painter.setBrush(brush)
            
            painter.setFont(font_info)
            font_size = QFontMetrics(font_info)
            
            
            if points_dict:
                for key, pt in points_dict.items():
                    if not pt.absence:
                        bbox = pt.rect
                        painter.drawEllipse(int(bbox.center().x()/(scale)),int(bbox.center().y()/(scale)),
                                            point_size.width(), point_size.height())
                        
                        #get the point name
                        painter.drawText(int(bbox.center().x()//(scale) - font_size.width(pt.pt_name)//2),
                                         int(bbox.center().y()//(scale) - font_size.height()//2),
                                         pt.pt_name)


    def draw_seg_args(self, segments_cv):
        """Draw segmentation based on the scale (original reso / thumbnail reso)
        Use the function draw_seg_cv(self,img_cv_draw,contour_cv,color,scale)
            
        Args:
            segments_cv (_type_): A dictionary of segmentation
        """   

        scale = max(self.data.current_pixmap.width()/thumbnail_size.width(), self.data.current_pixmap.height()/thumbnail_size.height())

        height = int(self.data.get_current_origin_pixmap().height() /scale)
        width = int(self.data.get_current_origin_pixmap().width() /scale)

        # img_cv_draw is the empty mask (with zeros) 
        # scaled zero
        img_cv_draw = np.zeros((height, width ,4)).astype('uint8')
        img_cv_draw = cv2.cvtColor(img_cv_draw,cv2.COLOR_BGRA2RGBA)

        # Draw the segmentation using draw_seg_cv(self,img_cv_draw,contour_cv,color,scale)
        if segments_cv:
            for _, (key, item) in enumerate(segments_cv.items()):
                contour_cv = item['contours']
                segs_name_id_map = self.data.get_current_image_seg_map()
                idx_item = segs_name_id_map[key]
                cur_color = self.parent().window().seg_colours[idx_item%len(self.parent().window().seg_colours)]
                self.draw_seg_cv(img_cv_draw,contour_cv , cur_color , scale)



        image_cv_draw = QImage(img_cv_draw, img_cv_draw.shape[1],\
        img_cv_draw.shape[0], img_cv_draw.shape[1] * 4,QImage.Format_RGBA8888)

        return QPixmap(image_cv_draw)
        # self.canvas  = QPixmap(image_cv_draw)



    def draw_seg_cv(self,img_cv_draw,contour_cv,color,scale):
        """Draw opencv image using contours

        Args:
            img_cv_draw (_type_): Image (mask)
            contour_cv (_type_): Contour in OpenCV format
            color (_type_): The colour of the mask
            scale (_type_): thumbnail scale
        """          
   
        if contour_cv is not None:
            contour_cv = [(np.array(contour, dtype='int32') //scale).astype('int32') for contour in contour_cv]
            cv_colour = (color.red() , color.green() , color.blue() ,color.alpha())
            # print("cv colour in Browse_Panel.draw_seg_cv" , cv_colour)
            cv2.fillPoly(img_cv_draw, contour_cv, cv_colour)
            # img_cv_draw[img_cv_draw[...,3] >0 , : ] = cv_colour



    def prepare_images(self):
        """ Action of visualiasing when scrolling. Using multi-threads to increase the view speed.

        """

        scroll_bar = self.widget_image_browser.verticalScrollBar()
        # Calculate images to be shown in here:
        image_count = self.widget_image_browser.count()

        if image_count!=0:
            # Calculate the current images count for the view port
            width = self.widget_image_browser.rect().size().width()
            images_per_row = floor(width/(self.widget_image_browser.item(0).sizeHint().width() + self.widget_image_browser.spacing()))


            height = self.widget_image_browser.rect().size().height()
            images_per_col =ceil(height / (self.widget_image_browser.item(0).sizeHint().height() + self.widget_image_browser.spacing()))

            start = scroll_bar.value()
            prev_rows = ceil(start/ (self.widget_image_browser.item(0).sizeHint().height() + self.widget_image_browser.spacing()))
            prev_imgs = prev_rows * images_per_row




            start_img = prev_imgs -(images_per_row)
            start_img_id = max(0,start_img)
            end_img_id = min(image_count , start_img_id+images_per_row*(images_per_col+1))

            # Check the thumbnail cache and decide to clean cache. The thumbnail cache limit is 100

            thumbnail_list_size_limit =100
            thumbnail_idx_list = [idx for idx,thumbnail in enumerate(self.thumbnail_list) if thumbnail is not None]
            if len(thumbnail_idx_list)>thumbnail_list_size_limit:
                length = len(thumbnail_idx_list)
                emptied_count=0
                for idx in thumbnail_idx_list:
                    if idx < start_img or idx > end_img_id:
                        self.thumbnail_list[idx] = None
                        self.widget_image_browser.item(idx).setIcon(QIcon())
                        emptied_count+=1

                    if length - emptied_count<thumbnail_list_size_limit:
                        break


            # get a list of image names that should be displayed
            img_list = [self.data.images[str(self.widget_image_browser.item(img_id).text())] for img_id in range(start_img_id, end_img_id)]
            img_id_list = [img_id for img_id in range(start_img_id, end_img_id)]

            # if self.worker is not None:
            #     self.threadpool.cancel(self.worker)
            #
        
        
            #### non-multi threads add images for testing  #####
           
            for image, img_id in zip(img_list , img_id_list):
                points_dict = image.points_dict
                segments_cv = image.segments_cv

                if self.widget_image_browser.item(img_id).icon().isNull():
                # If the icon is NUll, read images

                # read images and draw annotation
                # Read images
                    pixmap = QPixmap(os.path.join(self.data.work_dir, image.img_name))
                    pixmap = pixmap.scaled(thumbnail_size, aspectRatioMode=Qt.KeepAspectRatio)
                else:
                    pixmap = None

                self.add_image([img_id, pixmap , points_dict, segments_cv])

            
            #### multi threads version of adding images ######
            
            # self.worker = Worker_read_thumbnail(self.add_image_QRunnable, img_list=img_list ,
            #                                     img_id_list=img_id_list)
            # self.threadpool.start(self.worker)


            




    def resizeEvent(self, e):
        self.prepare_images()

    def trigger_check_state(self,item):
        """
        Put flag on images

        :param item:
        :return:
        """
        try:
            # current_row = self.widget_image_browser.row(item)
            current_name = str(item.text())
            state = item.checkState()
            if  state== 0:
                self.data.images[current_name].attention_flag = False
            elif state == 2:
                self.data.images[current_name].attention_flag = True
            else:
                raise ValueError("check state error: state value:{}".format(state))
        except ValueError as ve:
            print(ve)

    def go_to_annotation(self, item):
        current_row = self.widget_image_browser.row(item)
        if self.parent():
            widget_file_list = self.parent().window().widget_file_list
            widget_file_list.setCurrentRow(current_row)

            self.parent().window().act_browse_mode.activate(QAction.Trigger)


    ## functions that are not using now ####
    def draw_points_args(self ,points_dict):
        """Draw keypoint based on the scale (original reso / thumbnail reso)

        Args:
            points_dict (_type_): A dictionary of points detail
        """        


        scale = max(self.data.current_pixmap.width()/thumbnail_size.width(), self.data.current_pixmap.height()/thumbnail_size.height())

        pen = QPen(Qt.red, 2)
        brush = QBrush(QColor(0, 255, 255, 120))
        self.painter.setPen(pen)
        self.painter.setBrush(brush)

        if points_dict:
            for key, pt in points_dict.items():
                if not pt.absence:
                    bbox = pt.rect
                    self.painter.drawEllipse(bbox.center()/(scale), point_size.width(), point_size.height())
                    self.painter.drawText(bbox.center()/(scale), pt.pt_name)

class Worker_read_thumbnail(QRunnable):
    """Class used to draw thumbnail images for review mode.

    Args:
        QRunnable (_type_)
    """


    def __init__(self, fn, **kwargs):
        super(Worker_read_thumbnail, self).__init__()
        # Store constructor arguments (re-used for processing)
        self.fn = fn
        #self.args = args
        self.kwargs = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
    @pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''

        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(
                img_list = self.img_list , img_id_list = self.img_id_list
            )
        except Exception as e:
            print(e)
        #     exctype, value = sys.exc_info()[:2]
        #     self.signals.error.emit((exctype, value, e))
        # else:
        #     self.signals.result.emit(result)  # Return the result of the processing
        # finally:
        #     self.signals.finished.emit()  # Done
class ReadThumbnail_object(QObject):
    finished = pyqtSignal()

    def __init__(self, **kwargs):
        """
        Initiate
        window, img_lsit, img_id_lsit, workdir
        """
        super(ReadThumbnail_object, self).__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.err = None

    @pyqtSlot()
    def add_image(self):
        try:
            for image, img_id in zip(self.img_list , self.img_id_list):
                points_dict = image.points_dict
                segments_cv = image.segments_cv

                if self.widget_image_browser.item(img_id).icon().isNull():
                # If the icon is NUll, read images

                # read images and draw annotation
                # Read images
                    pixmap = QPixmap(os.path.join(self.work_dir, image.img_name))
                    pixmap = pixmap.scaled(thumbnail_size, aspectRatioMode=Qt.KeepAspectRatio)
                else:
                    pixmap = None


                img_name_id = self.window.data.img_id_order[img_id]
                # print(img_id, pixmap is not None , self.data.images[img_id].label_changed)
                if not (pixmap is None and self.window.data.images[img_name_id].label_changed == False):
                    if pixmap is not None:
                        self.window.thumbnail_list[img_id] = pixmap.copy()
                    elif self.window.data.images[img_name_id].label_changed:
                        pixmap = self.window.thumbnail_list[img_id].copy()
                        self.window.data.images[img_name_id].label_changed = False

                    self.window.painter.begin(pixmap)
                    self.window.draw_points_args(points_dict)

                    self.window.painter.drawPixmap(0,0,self.window.draw_seg_args(segments_cv))

                    self.window.painter.end()

                    pixmap = pixmap.scaled(icon_size, aspectRatioMode=Qt.KeepAspectRatio)
                    icon = QIcon()
                    icon.addPixmap(pixmap, QIcon.Normal, QIcon.Off)
                    self.window.widget_image_browser.item(img_id).setIcon(icon)
        except Exception as e:
            self.err=e
            print(e)

        self.finished.emit()


class ReadThumbnailThread(QThread):
    read_one_image = pyqtSignal([list])

    def __init__(self, img_list=[] , img_id_list=[] , work_dir = None):
        QThread.__init__(self)
        self.img_list = img_list
        self.img_id_list = img_id_list
        self.work_dir = work_dir



    def __del__(self):
        self.wait()

    def update_img_list(self, img_list, img_id_list, widget_image_browser):
        self.img_list = img_list
        self.img_id_list = img_id_list
        self.widget_image_browser = widget_image_browser

    def run(self):
        for image, img_id in zip(self.img_list , self.img_id_list):

            points_dict = image.points_dict
            segments_cv = image.segments_cv

            if self.widget_image_browser.item(img_id).icon().isNull():
            # If the icon is NUll, read images

            # read images and draw annotation
            # Read images
                pixmap = QPixmap(os.path.join(self.work_dir, image.img_name))
                pixmap = pixmap.scaled(thumbnail_size, aspectRatioMode=Qt.KeepAspectRatio)
            else:
                pixmap = None

            self.read_one_image.emit([img_id, pixmap, points_dict,segments_cv])

            # self.sleep(1)

if __name__ == '__main__':
    start = timeit.default_timer()

    app = QApplication(sys.argv)
    ex = BrowsePanel()

    stop = timeit.default_timer()
    ex.show()
    print('Time: ', stop - start)

    sys.exit(app.exec_())
