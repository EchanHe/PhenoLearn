from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys,os
import relabelData

from math import ceil, floor
import cv2


import timeit

thumbnail_size = QSize(350,350)
point_size = QSize(5,5)
class BrowsePanel(QWidget):

    def __init__(self,  data = None):
        super().__init__()

        self.initUI(data = data)


    def initUI(self, data ):
        if data is not None:
            self.data = data
        self.layout = QVBoxLayout(self)

        self.widget_image_browser = QListWidget()
        self.widget_image_browser.setIconSize(thumbnail_size)

        self.widget_image_browser.setViewMode(QListView.IconMode)


        self.widget_image_browser.setFlow((QListView.LeftToRight))
        self.widget_image_browser.setResizeMode(QListView.Adjust)

        self.widget_image_browser.setSpacing(10)

        self.widget_image_browser.verticalScrollBar().valueChanged.connect(self.prepare_images)

        # listWidget->setResizeMode(QListView::Adjust);
        # listWidget->setGridSize(QSize(64, 64));
        #
        # listWidget->setSpacing(someInt);


        self.layout.addWidget(self.widget_image_browser)
        self.setLayout(self.layout)

        self.init_for_testing()
        self.init_list()
        # self.setMouseTracking(True)
        # self.resize.connect(self.sizechanged)

        self.setGeometry(300, 300, 800, 600)
        self.setWindowTitle('Browse Panel')
        self.show()

        self.setMouseTracking(True)

    def init_for_testing(self):
        # Used for testing the widget itself.
        # file_name = 'data/data_file.json'
        # work_dir = './data/'

        file_name = 'genus.json'
        work_dir = '../plumage/data/thumbnail/'

        self.data = relabelData.Data(file_name,work_dir)

        # self.pixmap = QPixmap(os.path.join(self.data.work_dir, self.data.images[0].img_name))
        # self.orignal_size = self.pixmap.size()
        # self.pixmap = self.pixmap.scaled(thumbnail_size, aspectRatioMode=Qt.KeepAspectRatio)
        # self.state_browse = True

    def init_list(self):
        images = self.data.images
        for image in images:
            # QT part
            # item = QListWidgetItem(image.img_name)

            icon = QIcon()



            item = QListWidgetItem()
            item.setSizeHint(thumbnail_size)
            item.setIcon(icon)
            self.widget_image_browser.addItem(item)

        # # Init first images:
        # for idx, image in enumerate(images[:4]):
        #     self.points = image.points
        #
        #     # read images and draw annotation
        #
        #     self.pixmap = QPixmap(os.path.join(self.data.work_dir, image.img_name))
        #     self.orignal_size = self.pixmap.size()
        #     self.pixmap = self.pixmap.scaled(thumbnail_size, aspectRatioMode=Qt.KeepAspectRatio)
        #     self.painter = QPainter()
        #
        #     self.painter.begin(self.pixmap)
        #     self.draw_points()
        #     self.painter.end()
        #
        #     icon.addPixmap(self.pixmap, QIcon.Normal, QIcon.Off)


    def draw_points(self):
                # From the scale point , draw points
        scale = max(self.orignal_size.width()/thumbnail_size.width(), self.orignal_size.height()/thumbnail_size.height())
        # self.painter.scale(1/scale,1/scale)

        # self.painter.scale(1/scale,1/scale)

        pen = QPen(Qt.red, 2)
        brush = QBrush(QColor(0, 255, 255, 120))
        self.painter.setPen(pen)
        self.painter.setBrush(brush)


        if self.points:
            for pt in self.points:
                if not pt.absence:
                    bbox = pt.rect
                    self.painter.drawEllipse(bbox.center()/10*scale, point_size.width(), point_size.height())
                    # self.painter.drawEllipse(100,100, bbox.width()//2, bbox.height()//2)

    def mouseMoveEvent(self, e):
        return None
        # print(self.widget_image_browser.size(), self.widget_image_browser.rect().size(),self.widget_image_browser.spacing())
        # print(self.widget_image_browser.item(0).sizeHint(), self.widget_image_browser.iconSize())



        # width = self.widget_image_browser.rect().size().width() - self.widget_image_browser.spacing()-1
        # print(width/(self.widget_image_browser.item(0).sizeHint().width() + self.widget_image_browser.spacing()))
        # height = self.widget_image_browser.rect().size().height() - self.widget_image_browser.spacing()
        # print(height / (self.widget_image_browser.item(0).sizeHint().height() + self.widget_image_browser.spacing()))

        # print(self.widget_image_browser.visibl)


    def prepare_images(self):
        scroll_bar = self.widget_image_browser.verticalScrollBar()



        # Calculate images to be shown in here:
        image_count = len(self.data.images)

        # Calculate the current images count for the view port
        width = self.widget_image_browser.rect().size().width()
        images_per_row = floor(width/(self.widget_image_browser.item(0).sizeHint().width() + self.widget_image_browser.spacing()))


        height = self.widget_image_browser.rect().size().height()
        images_per_col =ceil(height / (self.widget_image_browser.item(0).sizeHint().height() + self.widget_image_browser.spacing()))

        img_per_viewport = images_per_row* images_per_col

        start = scroll_bar.value()
        prev_rows = ceil(start/ (self.widget_image_browser.item(0).sizeHint().height() + self.widget_image_browser.spacing()))
        prev_imgs = prev_rows * images_per_row

        print(images_per_row, images_per_col)
        print("total images:", img_per_viewport)
        print("rows:",prev_rows)
        print("imgs before:", prev_imgs)


        # # Clear all images
        # if prev_imgs-images_per_row >= 0:
        #     for i in range(prev_imgs-images_per_row, prev_imgs):
        #         icon = QIcon()
        #         self.widget_image_browser.item(i).setIcon(icon)

        # Read and load imgs:

        start_img = img_per_viewport + prev_imgs - images_per_row

        start_img_id = max(0,start_img-images_per_row)
        end_img_id = min(image_count , start_img+images_per_row)
        print(start_img_id, end_img_id)
        for img_id in range(start_img_id, end_img_id):

            image = self.data.images[img_id]
            self.points = image.points

            # read images and draw annotation

            self.pixmap = QPixmap(os.path.join(self.data.work_dir, image.img_name))
            # self.pixmap = QPixmap("small.jpg")
            self.orignal_size = self.pixmap.size()
            self.pixmap = self.pixmap.scaled(thumbnail_size, aspectRatioMode=Qt.KeepAspectRatio)
            self.painter = QPainter()

            self.painter.begin(self.pixmap)
            self.draw_points()
            self.painter.end()

            icon = QIcon()
            icon.addPixmap(self.pixmap, QIcon.Normal, QIcon.Off)
            self.widget_image_browser.item(img_id).setIcon(icon)




    # def sizechanged(self):
    #     print(self.rect().size())

    def resizeEvent(self, e):
        print(self.rect().size())
        self.prepare_images()
    # def resizeEvent(self, e):
    #
    #     # Calculate the image
    #     scroll_bar = self
    #     scroll_bar.value()
    #
    #
    # def wheelEvent(self,e):
    #     scroll_bar = self.widget_image_browser.verticalScrollBar()
    #
    #     print(scroll_bar.value())

if __name__ == '__main__':
    start = timeit.default_timer()

    app = QApplication(sys.argv)
    ex = BrowsePanel()

    stop = timeit.default_timer()

    print('Time: ', stop - start)

    sys.exit(app.exec_())
