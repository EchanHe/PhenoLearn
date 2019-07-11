from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys,os
import relabelData

from math import ceil, floor
import cv2


import timeit

item_size = QSize(350,350)
thumbnail_size = QSize(300,300)
point_size = QSize(5,5)
class BrowsePanel(QWidget):

    def __init__(self,  data = None):
        super().__init__()

        self.initUI(data = data)

    # def __init__(self, init=True):
    #     super().__init__()
    #     self.init_for_testing()
    #     self.initUI(data = None)

    def initUI(self, data ):
        if data is not None:
            self.data = data
        self.layout = QVBoxLayout(self)

        self.widget_image_browser = QListWidget()
        self.widget_image_browser.setIconSize(item_size)

        self.widget_image_browser.setViewMode(QListView.IconMode)

        self.widget_image_browser.setFlow((QListView.LeftToRight))
        self.widget_image_browser.setResizeMode(QListView.Adjust)
        self.widget_image_browser.setSpacing(10)

        self.widget_image_browser.setMovement(QListView.Static)

        self.widget_image_browser.verticalScrollBar().valueChanged.connect(self.prepare_images)
        self.widget_image_browser.itemClicked.connect(self.trigger_check_state)
        self.widget_image_browser.itemDoubleClicked.connect(self.go_to_annotation)


        self.layout.addWidget(self.widget_image_browser)
        self.setLayout(self.layout)

        # self.init_for_testing()
        self.init_list()
        self.thumbnail_dir = '../plumage/data/thumbnail/'

        # self.setMouseTracking(True)
        # self.resize.connect(self.sizechanged)

        self.setGeometry(300, 300, 800, 600)
        self.setWindowTitle('Browse Panel')
        # self.show()

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
        """
        Init list item without icons
        :return:
        """
        self.widget_image_browser.clear()
        images = self.data.images
        for image in images:
            # QT part
            item = QListWidgetItem()
            item.setSizeHint(item_size)
            # item.setIcon(icon)
            item.setText(image.img_name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            if image.attention_flag:
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)
            self.widget_image_browser.addItem(item)

    def reset_widget(self):
        self.init_list()

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
                    self.painter.drawEllipse(bbox.center()/(20*scale), point_size.width(), point_size.height())
                    # self.painter.drawEllipse(100,100, bbox.width()//2, bbox.height()//2)




    def prepare_images(self):
        scroll_bar = self.widget_image_browser.verticalScrollBar()
        # Calculate images to be shown in here:
        image_count = len(self.data.images)

        if image_count!=0:
            # Calculate the current images count for the view port
            width = self.widget_image_browser.rect().size().width()
            images_per_row = floor(width/(self.widget_image_browser.item(0).sizeHint().width() + self.widget_image_browser.spacing()))


            height = self.widget_image_browser.rect().size().height()
            images_per_col =ceil(height / (self.widget_image_browser.item(0).sizeHint().height() + self.widget_image_browser.spacing()))

            img_per_viewport = images_per_row* images_per_col

            start = scroll_bar.value()
            prev_rows = ceil(start/ (self.widget_image_browser.item(0).sizeHint().height() + self.widget_image_browser.spacing()))
            prev_imgs = prev_rows * images_per_row


            # # Clear all images
            # if prev_imgs-images_per_row >= 0:
            #     for i in range(prev_imgs-images_per_row, prev_imgs):
            #         icon = QIcon()
            #         self.widget_image_browser.item(i).setIcon(icon)

            # Read and load imgs:

            start_img = prev_imgs -(images_per_row*images_per_col)
            start_img_id = max(0,start_img)
            end_img_id = min(image_count , start_img_id+images_per_row*(images_per_col+2))
            for img_id in range(start_img_id, end_img_id):

                image = self.data.images[img_id]
                self.points = image.points

                # read images and draw annotation

                self.pixmap = QPixmap(os.path.join(self.thumbnail_dir, image.img_name))
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
        print("window size:",self.rect().size())
        self.prepare_images()

    def trigger_check_state(self,item):
        """
        Put attention flag on image

        :param item:
        :return:
        """
        try:
            current_row = self.widget_image_browser.row(item)
            state = item.checkState()
            if  state== 0:
                self.data.images[current_row].attention_flag = False
            elif state == 2:
                self.data.images[current_row].attention_flag = True
            else:
                raise ValueError("check state error: state value:{}".format(state))
        except ValueError as ve:
            print(ve)

    def go_to_annotation(self, item):
        current_row = self.widget_image_browser.row(item)
        if self.parent():
            widget_file = self.parent().window().widget_file_list
            widget_file.setCurrentRow(current_row)

            self.parent().window().act_browse_mode.activate(QAction.Trigger)



if __name__ == '__main__':
    start = timeit.default_timer()

    app = QApplication(sys.argv)
    ex = BrowsePanel(init = True)

    stop = timeit.default_timer()
    ex.show()
    print('Time: ', stop - start)

    sys.exit(app.exec_())
