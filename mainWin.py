#!/usr/bin/python3
# -*- coding: utf-8 -*-


from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import sys
import label_panel
import os
from pathlib import Path
import json
import glob
import datetime
from shutil import copyfile

from relabelData import Data

PROGRAM_NAME = 'Relabelling'


#pre-setting for reading json data
name_col = 'file_name'
point_col = "points"

prop_names = ['pt name' , 'x' , 'y']





def create_table_item(value, editbale = True):
    temp_item = QTableWidgetItem()
    item_flag = temp_item.flags()

    temp_item.setData(Qt.EditRole, value)
    if not editbale:
        temp_item.setFlags(item_flag & ~Qt.ItemIsEditable)
    return temp_item

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        # super(MainWindow, self).__init__()
        self.initUI()

    def init_load_files_test(self):
        self.file_name = 'genus.json'
        self.work_dir = '../plumage/data/vis/'

        self.data = Data(self.file_name,self.work_dir)

        self.list_file_names()


    def initUI(self):

        self.data_file_name = None
        self.data = None
        self.extension = None

        self.all_file_list = []

        self.file_path = '.'
        self.work_dir = None
        self.current_file_name = None

        #annoations
        self.current_x = None
        self.current_y = None
        self.current_pt_name = None
        self.current_pt_list= None

        self.statusBar().showMessage(PROGRAM_NAME)
        self.statusBar().show()

        self.file_dock = QDockWidget('Files', self)
        self.property_dock = QDockWidget('Annotations', self)
        self.addDockWidget(Qt.RightDockWidgetArea, self.property_dock)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.file_dock)

        # File list part
        self.widget_file_list = QListWidget()
        # self.widget_file_list.itemClicked.connect(self.file_list_click)
        # self.widget_file_list.itemDoubleClicked.connect(self.file_list_db_click)
        self.widget_file_list.currentItemChanged.connect(self.file_list_current_item_changed)
        self.widget_folder_label = QLabel(" Working Dir\n")

        self.widget_anno_file_label = QLabel("Annotation\n")

        layout_file_dock = QVBoxLayout()
        layout_file_dock.setContentsMargins(0, 0.1, 0, 0.1)
        layout_file_dock.addWidget((self.widget_anno_file_label))
        layout_file_dock.addWidget(self.widget_folder_label)
        layout_file_dock.addWidget(self.widget_file_list)
        widget_file_dock = QWidget()
        widget_file_dock.setLayout(layout_file_dock)
        self.file_dock.setWidget(widget_file_dock)

        # Annotation part, tabs

        self.widget_anno_tabs = QTabWidget()




        self.widget_point_list = QListWidget()
        self.widget_point_list.currentItemChanged.connect(self.point_list_current_item_changed)

        self.widget_contour_list = QListWidget()

        self.widget_anno_tabs.addTab(self.widget_point_list, "points")
        self.widget_anno_tabs.addTab(self.widget_contour_list, "outlines")

        #properties part
        self.widget_props_table = QTableWidget(0,1)
        self.widget_props_table.itemChanged.connect(self.prop_table_item_changed)
        # self.widget_props_table.itemDoubleClicked.connect(self.prop_table_db_click)
        # self.widget_props_table.currentItemChanged.connect(self.item_changed)

        self.widget_props_table.horizontalHeader().setVisible(False)
        # self.widget_props_table.setVerticalHeaderLabels(prop_names)

        layout_prop_dock = QVBoxLayout()

        # layout_prop_dock.setContentsMargins(0, 0.1, 0, 0.1)
        layout_prop_dock.addWidget(self.widget_anno_tabs)
        layout_prop_dock.addWidget(self.widget_props_table)

        widget_prop_dock = QWidget()
        widget_prop_dock.setLayout(layout_prop_dock)
        self.property_dock.setWidget(widget_prop_dock)



        #status bar
        self.label_xy_status = QLabel('')
        self.statusBar().addPermanentWidget(self.label_xy_status)



        self.init_load_files_test()



        self.widget_annotation = label_panel.LabelPanel(self.data)
        self.scroll_area = QScrollArea()
        self.scroll_area.setBackgroundRole(QPalette.Dark)
        self.scroll_area.setWidget(self.widget_annotation)
        self.scroll_area.viewport().installEventFilter(self)
        # self.scroll_area.setWidgetResizable(True)
        # self.scroll_area.setVisible(False)

        self.init_action()
        self.init_menu()

        self.setCentralWidget(self.scroll_area)
        self.setGeometry(100, 100, 1200, 800)
        self.setWindowTitle(PROGRAM_NAME)
        self.show()




    def init_action(self):
        self.act_opendir = QAction("&Open image dir", self, triggered=self.opendir)
        self.act_open_annotations = QAction("Open &label file", self, triggered=self.open_annotations)

        self.act_save = QAction("&Save", self , shortcut="Ctrl+S", triggered=self.save_annotations)
        self.act_save_as = QAction("Save as", self, triggered=self.save_as)


        self.act_exit = QAction("E&xit", self, shortcut="Ctrl+Q", triggered=self.close)

        # self.zoomInAct = QAction("Zoom &In (25%)", self, shortcut="Ctrl++", enabled=False, triggered=self.zoomIn)
        # self.zoomOutAct = QAction("Zoom &Out (25%)", self, shortcut="Ctrl+-", enabled=False, triggered=self.zoomOut)
        # self.normalSizeAct = QAction("&Normal Size", self, shortcut="Ctrl+S", enabled=False, triggered=self.normalSize)
        # self.fitToWindowAct = QAction("&Fit to Window", self, enabled=False, checkable=True, shortcut="Ctrl+F",
        #                               triggered=self.fitToWindow)
        # self.aboutAct = QAction("&About", self, triggered=self.about)
        # self.aboutQtAct = QAction("About &Qt", self, triggered=qApp.aboutQt)
        #
        # self.draw_act = QAction("&draw rect", self, enabled=True, triggered=self.draw_rect)

    def init_menu(self):
        # File part
        self.menu_file = QMenu("&File", self)
        self.menu_file.addAction(self.act_open_annotations)
        self.menu_file.addAction(self.act_opendir)
        self.menu_file.addSeparator()
        self.menu_file.addAction(self.act_save)
        self.menu_file.addAction(self.act_save_as)
        self.menu_file.addSeparator()
        self.menu_file.addAction(self.act_exit)

        self.menuBar().addMenu(self.menu_file)

        #Edit part
        self.menu_edit = QMenu("&Edit", self)

        self.menuBar().addMenu(self.menu_edit)
        #View part:
        self.menu_view = QMenu("&View", self)
        self.menu_view.addAction(self.widget_annotation.act_zoom_in)
        self.menu_view.addAction(self.widget_annotation.act_zoom_out)
        self.menu_view.addAction(self.widget_annotation.act_origin_size)

        self.menuBar().addMenu(self.menu_view)

    def opendir(self, _value=False, dirpath=None):


        defaultOpenDirPath = os.path.dirname(self.file_path) if self.file_path else '.'

        temp = (QFileDialog.getExistingDirectory(self,
                                                     'Open dir for images', defaultOpenDirPath,
                                                     QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks))

        if temp and temp != self.work_dir:

            self.work_dir = temp

            self.data.set_work_dir(temp)

            self.list_file_names()

    def open_annotations(self):
        options = QFileDialog.Options()
        # fileName = QFileDialog.getOpenFileName(self, "Open File", QDir.currentPath())
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Annotations', '.',
                                                  'Files (*.csv *.json)', options=options)
        if file_name:
            file_name = os.path.abspath(file_name)
            self.data.set_file_name(file_name)


        self.list_file_names()

    def save_annotations(self):


        # Save the old version of file
        json_name = os.path.basename(self.data.file_name)
        json_dir = os.path.dirname(self.data.file_name)
        time_now = datetime.datetime.now()
        temp_dir = os.path.join( os.path.join(json_dir,'temp'), time_now.strftime("%Y-%m-%d_%H-%M-%S"))
        print(temp_dir)
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        temp_path = os.path.join(temp_dir,json_name)

        copyfile(self.data.file_name,temp_path)

        # Save the new data into the same name
        self.data.write_json()
        self.widget_anno_file_label.setText("Annotation file: {}".format(self.data.file_name))

    def save_as(self):

        current_dir = self.data.work_dir
        save_path, _ = QFileDialog.getSaveFileName(self, 'Saving Annotations',current_dir,"JSON (*.json)")
        self.data.write_json(save_path)
        self.widget_anno_file_label.setText("Annotation file: {}".format(self.data.file_name))

    def list_file_names(self):
        if self.data:
            self.widget_file_list.clear()

            if self.data.has_images():
                for img in self.data.images:
                    self.widget_file_list.addItem(img.img_name)

        work_dir = self.data.work_dir
        self.widget_folder_label.setText("Working Dir: {}".format(os.path.abspath(work_dir)))
        self.widget_anno_file_label.setText("Annotation file: {}".format(os.path.basename(self.data.file_name)))



    def list_properties(self):

        current_img = self.data.get_current_image()
        # Create props table
        if self.data.has_images():
            pt_props = self.data.get_current_image().get_curent_pt_props()
            self.widget_props_table.setRowCount(len(pt_props.keys()))
            self.widget_props_table.setVerticalHeaderLabels(pt_props.keys())
            for idx,prop in enumerate(pt_props):
                value = create_table_item(pt_props[prop])
                self.widget_props_table.setItem(idx,0, value)
        else:
            self.widget_props_table.setRowCount(0)
            self.widget_props_table.clear()

        # if current_img:
        #     pt_name  = create_table_item(current_img.get_current_pt_name())
        #     x = create_table_item(current_img.get_current_pt_x())
        #     y = create_table_item(current_img.get_current_pt_y())
        #
        #
        #     self.widget_props_table.setItem(0,0, pt_name)
        #     self.widget_props_table.setItem(1,0, x)
        #     self.widget_props_table.setItem(2,0, y)
        #
        # else:
        #     self.widget_props_table.clear()



    def list_point_name(self):
        self.widget_point_list.clear()

        points = self.data.get_current_image_points()
        if points:
            for pt in points:
                self.widget_point_list.addItem(pt.pt_name)

    def prop_table_item_changed(self, item):
        # Update data if any item in the property table changed

        if item is not None:
            print("asd")
            # Depending on different row of input.
            value = item.text()
            if self.widget_props_table.row(item) ==1:
                value=int(value)
                self.data.set_current_pt_of_current_img(x = value)
            if self.widget_props_table.row(item) ==2:
                value=int(value)
                self.data.set_current_pt_of_current_img(y=value)

            if self.data.changed:
                self.widget_anno_file_label.setText("Annotation file: {}*".format(self.data.file_name))
            self.widget_annotation.update()

    def file_list_current_item_changed(self,current,prev):

        if current:
            print("prepare images and annos")
            idx = self.widget_file_list.currentRow()
            self.widget_file_list.setCurrentRow(idx)

            self.data.set_image_id(idx)

            self.widget_annotation.update()

            # # Open image
            # self.widget_annotation.open_image(self.data.get_current_image_abs_path())
            # # Open annotations and Clean the state
            # self.widget_annotation.set_annotations(self.data.get_current_image_points())

        # # Set table:
        self.list_properties()
        self.list_point_name()


        # # Highlight the first row in the begining
        # if prev is None:
        #     self.widget_file_list.setCurrentRow(0)

    def point_list_current_item_changed(self,current,prev):
        print('change')
        idx_pt = self.widget_point_list.currentRow()
        self.widget_point_list.setCurrentRow(idx_pt)

        self.data.get_current_image().set_current_pt_id(idx_pt)
        self.list_properties()


    def eventFilter(self, source, event):

        modifiers = QApplication.keyboardModifiers()

        if (event.type() == QEvent.Wheel and source is self.scroll_area.viewport()):
            if modifiers and modifiers == Qt.ControlModifier:
                return True


        return False


if __name__ == '__main__':

    # fileName = 'data/data_file.json'
    # if fileName:
    #     _, extension  = os.path.splitext(fileName)
    #     if extension  =='.json':
    #         with open(fileName, "r") as read_file:
    #             data = json.load(read_file)
    #
    # d = Data(data,'data')
    # print(d.img_size)

    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())
