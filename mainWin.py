#!/usr/bin/python3
# -*- coding: utf-8 -*-


from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import sys
import label_panel
import browse_panel
import os
import pandas as pd
import datetime
from shutil import copyfile

from relabelData import Data_gui

# from deep_learning_tool import phenolearn_io


PROGRAM_NAME = 'Relabelling'

def create_table_item(value, editbale = True):
    """Create item for table

    Args:
        value (_type_): value of the item
        editbale (bool, optional): whether can be edited. Defaults to True.

    Returns:
        QTableWidgetItem: A QT table item with the value
    """
    temp_item = QTableWidgetItem()
    item_flag = temp_item.flags()

    temp_item.setData(Qt.EditRole, value)
    if not editbale:
        temp_item.setFlags(item_flag & ~Qt.ItemIsEditable)
    return temp_item

def iter_all_list_items(self):
    """iterate all items

    Yields:
        _type_: item value
    """
    for i in range(self.count()):
        yield self.item(i)

def iter_all_tab_widgets(self):
    
    for i in range(self.count()):
        yield self.tabText(i), self.widget(i)



class MainWindow(QMainWindow):
    """Main window of the app

    Args:
        QMainWindow (_type_): _description_
    """

    def __init__(self):
        super().__init__()
        # super(MainWindow, self).__init__()
        self.initUI()


    def initUI(self):
        """Init the UI part
        """
        
        #Function for init UI
        self.setStyleSheet( """ 
                                QListWidget:item:selected:!disabled {
                                 color:white;
                                     background: #75b7ff;
                                }
                                """
                                )

        self.seg_colours = [QColor(220,20,60,100), QColor(255,165,0,100),
                            QColor(238,232,170,102), QColor(255,255,0,103),
                            QColor(124,252,0,104), QColor(46,139,87,105),
                            QColor(0,255,255,106), QColor(0,0,255,107)]

        self.data = Data_gui()


        self.all_file_list = []

        self.file_path = '.'
        self.work_dir = None

        self.statusBar().showMessage(PROGRAM_NAME)
        self.statusBar().show()

        self.file_dock = QDockWidget('Files', self)
        self.property_dock = QDockWidget('Annotations', self)

        self.info_dock = QDockWidget('Quick label', self)


        self.info_dock.setVisible(False)

        self.addDockWidget(Qt.RightDockWidgetArea, self.property_dock)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.file_dock)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.info_dock)


        # Review assistant

        self.widget_review_assist = QWidget()

        self.widget_review_assist_combobox = QComboBox()


        self.button_review_sort = QPushButton('Sort')
        self.button_review_sort.clicked.connect(self.review_sort)
        self.buttone_review_reset= QPushButton('Reset')

        self.widget_review_tab = QTabWidget()

        # To be removed
        self.widget_review_properties = QListWidget()

        self.widget_review_properties.itemChanged.connect(self.widget_review_properties_item_click_filter)
        # end #


        layout = QVBoxLayout(self.widget_review_assist)
        layout.addWidget(QLabel("Review Assistant"))
        layout.addWidget(self.widget_review_assist_combobox)
        layout.addWidget(self.button_review_sort)

        layout.addWidget(self.widget_review_tab)
        layout.addWidget(self.buttone_review_reset)

        # File list part
        self.widget_file_list = QListWidget()

        self.widget_file_list.currentRowChanged.connect(self.file_list_current_item_changed)

        self.scroll_folder_label = QScrollArea()

        # self.layout_folder_label = QVBoxLayout()

        self.widget_folder_label= QLabel("Image Dir: {}\nAnnotation file: {}".format("",""))

        self.widget_anno_file_label = QLabel("Annotation file: {}".format(self.data.file_name))
        self.scroll_folder_label.setWidget(self.widget_folder_label)

        layout_file_dock = QVBoxLayout()
        # layout_file_dock.setContentsMargins(0, 0.1, 0, 0.1)
        # layout_file_dock.addWidget(self.scroll_folder_label,3)

        # layout_file_dock.addWidget((self.widget_anno_file_label))
        # layout_file_dock.addWidget(self.widget_folder_label)
        layout_file_dock.addWidget(self.widget_file_list,60)

        layout_file_dock.addWidget(self.widget_review_assist,40)

        ### splitter
        # self.splitter_file = QSplitter(Qt.Vertical)
        # self.splitter_file.addWidget(self.widget_file_list)
        # self.splitter_file.addWidget(self.widget_review_assist)
        # self.splitter_file.setStretchFactor(1, 1)
        # # self.splitter_file.setSizes([100,200])
        # layout_file_dock.addWidget(self.splitter_file)

        widget_file_dock = QWidget()
        widget_file_dock.setLayout(layout_file_dock)
        self.file_dock.setWidget(widget_file_dock)


        #### Annotation Panel ####
        self.widget_anno_tabs = QTabWidget()

        # Point list panel
        self.button_del_pt = QPushButton('Remove')
        self.button_del_pt.clicked.connect(self.delete_point)
        self.widget_point_list = QListWidget()
        self.widget_point_list.currentRowChanged.connect(self.point_list_current_item_changed)

        self.widget_pt = QWidget()
        layout = QVBoxLayout(self.widget_pt)
        layout.addWidget(self.button_del_pt )
        layout.addWidget(self.widget_point_list )

        # segmentation list panel
        self.widget_segment = QWidget()
        self.widget_segment_list = QListWidget()
        self.widget_segment_list.itemChanged.connect(self.update_segment_drawing)
        self.widget_segment_list.currentRowChanged.connect(self.segment_list_changed)

        self.widget_segment_control = QWidget()
        # Add and delete
        self.button_add_seg = QPushButton('Add')
        self.button_del_seg = QPushButton('Remove')

        self.button_add_seg.clicked.connect(self.add_segmentation)
        self.button_del_seg.clicked.connect(self.delete_segmentation)



        layout = QHBoxLayout(self.widget_segment_control)
        layout.addWidget(self.button_add_seg )
        layout.addWidget(self.button_del_seg )


        layout = QVBoxLayout(self.widget_segment)
        layout.addWidget(self.widget_segment_control)
        layout.addWidget(self.widget_segment_list)

        self.widget_anno_tabs.addTab(self.widget_pt, "Point")
        self.widget_anno_tabs.addTab(self.widget_segment, "Segmentation")

        #### Properties editor
        self.widget_props_table = QTableWidget(0,1)
        self.widget_props_table.itemChanged.connect(self.prop_table_item_changed)

        self.widget_props_table.horizontalHeader().setVisible(False)


        ## Quick label widget
        self.widget_quick_label = QWidget()

        self.str_quick_pt = "Points:"
        self.str_quick_seg = "Segmentation\nclasses:"

        self.label_quick_pt = QLabel(self.str_quick_pt)
        self.label_quick_seg = QLabel(self.str_quick_seg)
        self.layout_quick_label = QHBoxLayout()

        self.layout_quick_label.addWidget(self.label_quick_pt)
        self.layout_quick_label.addWidget(self.label_quick_seg)
        self.widget_quick_label.setLayout(self.layout_quick_label)
        self.widget_quick_label.setVisible(False)
        # self.info_dock.setWidget(self.widget_quick_label)

        layout_prop_dock = QVBoxLayout()
        # layout_prop_dock.setContentsMargins(0, 0.1, 0, 0.1)
        layout_prop_dock.addWidget(self.widget_anno_tabs)
        layout_prop_dock.addWidget(self.widget_props_table)
        layout_prop_dock.addWidget(self.widget_quick_label)

        widget_prop_dock = QWidget()
        widget_prop_dock.setLayout(layout_prop_dock)
        self.property_dock.setWidget(widget_prop_dock)



        #status bar
        self.label_xy_status = QLabel('')

        self.statusBar().addPermanentWidget(self.label_xy_status)
        self.statusBar().addPermanentWidget(self.widget_folder_label)


        self.data.signal_data_changed.connect(self.update_file_label)
        self.data.signal_has_images.connect(self.update_menu_has_imgs)
        self.data.signal_has_undo.connect(self.update_menu_undo)


        self.widget_annotation = label_panel.LabelPanel(self.data)
        self.widget_browser = browse_panel.BrowsePanel(self.data)
        self.scroll_area = QScrollArea()
        self.scroll_area.setBackgroundRole(QPalette.Dark)
        self.scroll_area.setWidget(self.widget_annotation)

        self.widget_point_list.setContextMenuPolicy(Qt.CustomContextMenu)
        # self.widget_point_list.customContextMenuRequested.connect(self.menu_point_list)

        self.scroll_area.viewport().installEventFilter(self)
        self.widget_file_list.viewport().installEventFilter(self)
        self.widget_point_list.viewport().installEventFilter(self)
        self.widget_browser.widget_image_browser.viewport().installEventFilter(self)
        self.widget_annotation.installEventFilter(self)
        # self.scroll_area.setWidgetResizable(True)
        # self.scroll_area.setVisible(False)

        self.widget_stack = QStackedWidget()
        self.widget_stack.addWidget(self.scroll_area)
        self.widget_stack.addWidget(self.widget_browser)

        # self.widget_segment_list.itemClicked.connect(self.widget_annotation.update)


        self.init_action()
        self.init_menu()

        self.setCentralWidget(self.widget_stack)
        self.setGeometry(100, 100, 1400, 1000)
        self.setWindowTitle(PROGRAM_NAME)
        self.show()

        self.list_file_names()



        # icon = QPixmap(10,10)
        # icon.fill(self.seg_colours[0])
        # self.widget_annotation.setCursor(QCursor(icon))

    def init_action(self):
        self.act_opendir = QAction("&Open Image Directory", self, triggered=self.opendir)
        self.act_open_annotations = QAction("Open &Annotation File", self, triggered=self.open_annotations)
        # self.act_set_thumbnail_dir = QAction("Set the folder of icons", self, triggered=self.open_thumbnail_dir)

        self.act_save = QAction("&Save", self , shortcut="Ctrl+S", triggered=self.save_annotations)
        self.act_save_as = QAction("Save as", self, triggered=self.save_as)

        self.act_import_csv = QAction("Import as csv", self, triggered=self.import_csv)
        self.act_export_csv = QAction("Export as csv", self, triggered=self.export_csv)

        self.act_exit = QAction("E&xit", self, shortcut="Ctrl+Q", triggered=self.close)

        self.act_undo = QAction("&Undo",self, shortcut="Ctrl+Z", triggered=self.undo)

        self.act_browse_mode = QAction("Review Mode", self, shortcut="Ctrl+b", triggered=self.toggle_review_mode, checkable=True)
        self.act_quick_label_mode = QAction("Quick label", self, triggered=self.toggle_quick_label_mode, checkable=True)

        # self.act_point_mode = QAction("Pts", self, triggered=self.toggle_point_mode, checkable=True)
        # self.act_outline_mode = QAction("outline", self, triggered=self.toggle_seg_mode, checkable=True)

        self.act_point_mode = QAction("Point", self,  checkable=True)
        self.act_outline_mode = QAction("Seg", self, checkable=True)

        self.act_group_modes = QActionGroup(self)

        self.act_group_modes.addAction(QAction("View", self,  checkable=True, checked = True))
        self.act_group_modes.addAction(self.act_point_mode)
        self.act_group_modes.addAction(self.act_outline_mode)

        self.act_attention_imgs_only = QAction("Show Flag Images", self, triggered=self.toggle_flag_img, checkable=True)

        self.act_delete_point = QAction("Delete point", self, triggered=self.delete_point)


        self.act_sort_file_names = QAction("sort file", self, triggered = self.sort_file_names , checkable=True, enabled = False)
        self.act_sort_anno_names = QAction("sort annotations", self, triggered = self.sort_anno_names , checkable=True, enabled = False)

        self.act_brush_object = QAction("Draw", self, checkable=True)
        self.act_brush_erase = QAction("Erase", self, checkable=True)
        self.act_fill_mask = QAction("Auto Fill", self, triggered=self.auto_fill)

        self.act_brush_object.setChecked(True)

        self.act_brush_0 = QAction("Size_0", self, checkable=True)
        self.act_brush_1 = QAction("Size_1", self, checkable=True)
        self.act_brush_2 = QAction("Size_2", self, checkable=True)
        self.act_brush_3 = QAction("Size_3", self, checkable=True)

        self.act_brush_1.setChecked(True)

        self.act_group_brushes = QActionGroup(self)

        self.act_group_brushes.addAction(self.act_brush_0)
        self.act_group_brushes.addAction(self.act_brush_1)
        self.act_group_brushes.addAction(self.act_brush_2)
        self.act_group_brushes.addAction(self.act_brush_3)



        self.act_group_brush_cate = QActionGroup(self)
        self.act_group_brush_cate.addAction(self.act_brush_object)
        self.act_group_brush_cate.addAction(self.act_brush_erase)
        self.act_group_brush_cate.addAction(self.act_fill_mask)
    
    def init_menu(self):
        """Init the menu and add actions to menu
        """
        # File part
        self.menu_file = QMenu("&File", self)
        self.menu_file.addAction(self.act_open_annotations)
        self.menu_file.addAction(self.act_opendir)
        # self.menu_file.addAction(self.act_set_thumbnail_dir)
        self.menu_file.addSeparator()
        self.menu_file.addAction(self.act_save)
        self.menu_file.addAction(self.act_save_as)
        self.menu_file.addAction(self.act_import_csv)
        self.menu_file.addAction(self.act_export_csv)
        self.menu_file.addSeparator()
        self.menu_file.addAction(self.act_exit)

        self.menuBar().addMenu(self.menu_file)

        #Edit part, comment with the self.act_undo
        # self.menu_edit = QMenu("&Edit", self)
        # self.menu_edit.addAction(self.act_undo)
        #
        # self.menuBar().addMenu(self.menu_edit)
        #View part:
        self.menu_view = QMenu("&View", self)
        self.menu_view.addAction(self.act_browse_mode)
        self.menu_view.addSeparator()
        self.menu_view.addAction(self.widget_annotation.act_zoom_in)
        self.menu_view.addAction(self.widget_annotation.act_zoom_out)
        self.menu_view.addAction(self.widget_annotation.act_origin_size)
        self.menu_view.addSeparator()

        self.menu_view.addAction(self.property_dock.toggleViewAction())
        self.menu_view.addAction(self.file_dock.toggleViewAction())

        self.menuBar().addMenu(self.menu_view)

        self.menu_tool = QMenu("&Tool", self)
        self.menu_tool.addAction(self.act_sort_file_names)
        self.menu_tool.addAction(self.act_sort_anno_names)

        self.menuBar().addMenu(self.menu_tool)

        self.toolbar = self.addToolBar("Tool bars")

        self.toolbar.addActions(self.act_group_modes.actions())
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.act_browse_mode)
        self.toolbar.addAction(self.act_attention_imgs_only)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.act_quick_label_mode)

        self.toolbar_outline =QToolBar("Outline")
        self.toolbar_outline.addActions(self.act_group_brush_cate.actions())
        self.toolbar_outline.addSeparator()
        self.toolbar_outline.addActions(self.act_group_brushes.actions())

        self.addToolBarBreak()
        self.addToolBar(Qt.TopToolBarArea, self.toolbar_outline)
        
        
    def opendir(self, _value=False, dirpath=None):
        """Open the directory of images
        
        Triggered by act_opendir

        Args:
            _value (bool, optional): _description_. Defaults to False.
            dirpath (_type_, optional): _description_. Defaults to None.
        """
        try:
            # open a dialog about the file
            defaultOpenDirPath = os.path.dirname(self.file_path) if self.file_path else '.'

            temp = (QFileDialog.getExistingDirectory(self,
                                                         'Open dir for images', defaultOpenDirPath,
                                                         QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks))


            if temp and temp != self.work_dir:
                self.work_dir = temp
                self.data.set_work_dir(temp)
                self.list_file_names()
                self.list_review_assist()
                # self.widget_browser.reset_widget()
                self.widget_folder_label.setText("Image Dir: {}\nAnnotation file: {}".format(os.path.abspath(self.data.work_dir),os.path.abspath(self.data.file_name)))
                # self.widget_folder_label.setText("Image Dir: {}".format(os.path.abspath(self.data.work_dir)))
        except Exception as e:
            print(e)

    def open_annotations(self):
        """Open the annotation file

        """
        try:
            self.message_unsave()
            options = QFileDialog.Options()
            # fileName = QFileDialog.getOpenFileName(self, "Open File", QDir.currentPath())
            file_name, _ = QFileDialog.getOpenFileName(self, 'Open Annotations', '.',
                                                      'Files (*.json)', options=options)
            if file_name:
                file_name = os.path.abspath(file_name)
                _, extension = os.path.splitext(file_name)
                if extension == ".json":
                    self.data.set_file_name(file_name)
                    self.data.set_changed(False)


                if self.data.work_dir is None:
                    self.widget_folder_label.setText("Image Dir: {}\nAnnotation file: {}".format("",os.path.abspath(self.data.file_name)))
                else:
                    self.widget_folder_label.setText("Image Dir: {}\nAnnotation file: {}".format(os.path.abspath(self.data.work_dir),os.path.abspath(self.data.file_name)))

                # self.widget_browser.reset_widget()
            self.list_review_assist()
            self.list_file_names()

        except Exception as e:
            print(e)



    def undo(self):
        """
        Undo action
        Currently unable
        :return:
        """
        return None

        self.data.undo_act()

        self.list_point_name()
        # self.list_properties()
        self.widget_annotation.update()
    def save_annotations(self):
        """ Save file
        triggered by act_save
        
        :return:
        """

        # Save the old version of file
        # json_name = os.path.basename(self.data.file_name)
        # json_dir = os.path.dirname(self.data.file_name)
        # time_now = datetime.datetime.now()
        # temp_dir = os.path.join( os.path.join(json_dir,'temp'), time_now.strftime("%Y-%m-%d_%H-%M-%S"))
        #
        # # Save a backup to temp dir
        # if not self.data.no_anno_file:
        #     if not os.path.exists(temp_dir):
        #         os.makedirs(temp_dir)
        #     temp_path = os.path.join(temp_dir,json_name)
        #     copyfile(self.data.file_name,temp_path)
        # else:
        #     self.save_as()
        #     self.data.no_anno_file = False

        try:
            if self.data.no_anno_file:
                self.save_as()
                self.data.no_anno_file = False


            # Save the new data into the same name
            self.data.write_json()
            # self.widget_anno_file_label.setText("Annotation file: {}".format(self.data.file_name))
        except Exception as e:
            print(e)
            
            
    def save_as(self):
        """save file as
        triggered by act_save_as
        """
    
        if self.data:
            current_dir = self.data.work_dir
            save_path, _ = QFileDialog.getSaveFileName(self, 'Saving Annotations',current_dir,"JSON (*.json)")
            self.data.write_json(save_path)
            self.data.file_name = save_path
                

        # self.widget_anno_file_label.setText("Annotation file: {}".format(self.data.file_name))


    def import_csv(self):
        try:
            self.message_unsave()
            options = QFileDialog.Options()
            # fileName = QFileDialog.getOpenFileName(self, "Open File", QDir.currentPath())
            file_name, _ = QFileDialog.getOpenFileName(self, 'Import as csv', '.',
                                                      'Files (*.csv)', options=options)
            if file_name:
                file_name = os.path.abspath(file_name)
                _, extension = os.path.splitext(file_name)

                if extension =='.csv':
                    self.file_name_temp_for_csv = file_name
                    df_data = pd.read_csv(file_name)
                    self.show_input_csv_detail(df_data)

        except Exception as e:
            print(e)


    def is_str_list(self, s):
        try:
            result = eval(s)
            if type(result) is list:
                return True
            else:
                return False
        except:
            return False
    def show_input_csv_detail(self,df):
        print(df.columns)

        self.csv_window = QMainWindow()

        self.widget_dict = {}

        self.widget_csv_cols = QWidget()

        self.widget_scroll_csv_cols = QScrollArea()


        self.widget_scroll_csv_cols.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.widget_scroll_csv_cols.setWidgetResizable(True)
        self.widget_scroll_csv_cols.setWidget(self.widget_csv_cols)
        self.layout_csv_cols = QHBoxLayout()
        self.widget_csv_cols.setLayout(self.layout_csv_cols)
        column_choices = ['Exclude this column','File column', 'Point column', 'Segmentation column' , 'Property column']
        for col in df.columns:
            widget = QWidget()
            layout = QVBoxLayout()
            layout.addWidget(QLabel(col))


            self.widget_dict[col] = QComboBox()
            self.widget_dict[col].addItems(column_choices)

            if col =='file':
                self.widget_dict[col].setCurrentIndex(1)
            elif col.endswith("_x") or col.endswith("_y"):
                self.widget_dict[col].setCurrentIndex(2)
            elif self.is_str_list(df[col].values[0]):
                self.widget_dict[col].setCurrentIndex(3)



            layout.addWidget(self.widget_dict[col])

            widget.setLayout(layout)

            self.layout_csv_cols.addWidget(widget)

        ## Automatically guess the columns



        widget = QWidget()
        layout = QVBoxLayout()


        self.button_set_csv_cols = QPushButton("OK")
        self.button_set_csv_cols.clicked.connect(self.set_csv_cols)

        layout.addWidget(self.widget_scroll_csv_cols)
        layout.addWidget(self.button_set_csv_cols)
        widget.setLayout(layout)

        self.csv_window.setCentralWidget(widget)


        self.csv_window.setGeometry(100, 100, 1200, 300)
        self.csv_window.setWindowModality(Qt.ApplicationModal)
        self.csv_window.show()

        # self.csv_widget.set

    def set_csv_cols(self):
        coord_cols = []
        outline_cols =[]
        prop_cols =[]
        id_col ="file"

        for col,combo_box in self.widget_dict.items():
            combo_idx = combo_box.currentIndex()

            if combo_idx == 1:
                id_col = col
            elif combo_idx == 2:
                coord_cols.append(col)
            elif combo_idx == 3:
                outline_cols.append(col)
            elif combo_idx == 4:
                prop_cols.append(col)
        self.csv_window.close()



        self.data.set_file_name_csv(self.file_name_temp_for_csv, id_col,coord_cols,outline_cols,prop_cols )
        self.data.set_changed(False)


        if self.data.work_dir is None:
            self.widget_folder_label.setText("Image Dir: {}\nAnnotation file: {}".format("",os.path.abspath(self.data.file_name)))
        else:
            self.widget_folder_label.setText("Image Dir: {}\nAnnotation file: {}".format(os.path.abspath(self.data.work_dir),os.path.abspath(self.data.file_name)))

        self.list_review_assist()
        self.list_file_names()

    def export_csv(self):
        if self.data:
            current_dir = self.data.work_dir
            save_path, _ = QFileDialog.getSaveFileName(self, 'Export annotations as CSV',current_dir,"CSV (*.csv)")
            self.data.write_csv(save_path)


    def list_file_names(self):
        """List image names in the file panel.
        """

        if self.data:
            self.widget_file_list.clear()

            if self.data.has_images():
                # for img in self.data.images:
                #     self.widget_file_list.addItem(img.img_name)
                # for img_name,_ in self.data.images.items():
                #     self.widget_file_list.addItem(img_name)
                for img_name in self.data.img_id_order:
                    self.widget_file_list.addItem(img_name)

                self.widget_annotation.has_no_hidden = self.hide_file_names()



            # Update everything, if the file lists changed.
            self.widget_browser.reset_widget()


        self.list_properties()

        # self.widget_anno_file_label.setText("Annotation file: {}".format(os.path.basename(self.data.file_name)))


    def list_review_assist(self):
        """List specimen characteristics on Review assistant （review panel）
        The default is the Name
        """        
        self.widget_review_assist_combobox.clear()
        # list in review properties
        for key,item in self.data.img_props.items():
            if type(item[0]) is not str:

                self.widget_review_assist_combobox.addItem(key)
            else:
                props = sorted(list(set(item)))
                widget_list = QListWidget()
                widget_list.itemChanged.connect(self.widget_review_properties_item_click_filter)
                for prop in props:
                    item = QListWidgetItem()
                    item.setText(prop)
                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                    item.setCheckState(Qt.Checked)
                    widget_list.addItem(item)

                self.widget_review_tab.addTab(widget_list, key)



    def widget_review_properties_item_click_filter(self, item):
        """
        Filter image list when properties are unchecked
        """
        filtered_dict = {}
        for tab_text, widget in iter_all_tab_widgets(self.widget_review_tab):
            prop_key = str(tab_text)
            filtered_dict[prop_key] = []
            for item in iter_all_list_items(widget):
                if item.checkState():
                    filtered_dict[prop_key].append(item.text())

        flagged_img_idx = self.data.filter_imgs_by_review_assist(filtered_dict)

        if self.act_attention_imgs_only.isChecked():
            flagged_img_idx = list(set(flagged_img_idx) & set(self.data.flagged_img_idx))

        self.filter_img(flagged_img_idx, True)

    def review_sort(self):
        if self.widget_review_assist_combobox.count()>0:
            print(" sort images by files")
            self.data.sort_by_value(self.widget_review_assist_combobox.currentText())
            self.list_file_names()

    def list_point_name(self):
        """List names of points on the point panel
        
        Called when selected image changed, point added or deleted
        """

        self.widget_point_list.clear()
        points = self.data.get_current_image_points()
        if points is not None:
            keys = list(points.keys())
            if self.act_sort_anno_names.isChecked():
                # sort annotations

                keys.sort()
                print("sorted key in Mainwindow.list_point_name", keys)
            for key in keys:
                self.widget_point_list.addItem(key)

            cur_key = self.data.get_current_image().get_current_pt_key()

            if cur_key is not None and cur_key in points:
                keys = list(points.keys())
                idx = keys.index(cur_key)
                self.widget_point_list.setCurrentRow(idx)

    def list_seg_name(self):
        """List segmentations to the seg list widget in the annotation panel
        Combining segmentation name and colour information into a dict
        
        Called when selected image changed, seg added or deleted

        """
        self.current_image_colour_map = {}


        self.widget_segment_list.clear()
        # self.widget_segment_combobox.clear()

        segments = self.data.get_current_image_segments_cv()
        if segments is not None:
            keys = list(segments.keys())
            if self.act_sort_anno_names.isChecked():
                # sort annotaions
                keys.sort()

            for idx, key in enumerate(keys):
                item = QListWidgetItem()
                item.setText(key)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)

                item.setCheckState(Qt.Checked)

                icon = QPixmap(10,10)
                icon.fill(self.seg_colours[idx % len(self.seg_colours)])

                self.current_image_colour_map[key] = self.seg_colours[idx % len(self.seg_colours)]

                item.setIcon(QIcon(icon))

                self.widget_segment_list.addItem(item)

            self.widget_segment_list.setCurrentRow(0)
            self.update_segment_drawing()

    def list_properties(self):
        """
        List the currently selected label's properties

        :return:
        """

        # Use as locking prop_table_item_changed.


        self.prop_change_lock = True

        # Create props table
        if self.data.has_points_current_image():
            pt_props = self.data.get_current_image().get_curent_pt_props_dict()
            self.widget_props_table.setRowCount(len(pt_props.keys()))
            self.widget_props_table.setVerticalHeaderLabels(pt_props.keys())
            for idx,prop in enumerate(pt_props):
                value = create_table_item(pt_props[prop])
                self.widget_props_table.setItem(idx,0, value)
        else:
            self.widget_props_table.setRowCount(0)
            self.widget_props_table.clear()

        self.prop_change_lock = False

    def prop_table_item_changed(self, item):
        """
        Update data if any item in the property table changed
        :param item:

        """
        if item is not None and self.prop_change_lock == False:
            # Depending on different row of input.
            value = item.text()

            if value.lower() == 'true':
                 value = True
            elif value.lower() == 'false':
                 value =  False

            row = self.widget_props_table.row(item)
            key = self.widget_props_table.verticalHeaderItem(row).text()

            if key =='pt_name':
                changed = self.data.check_new_name_in_current_point_dict(value)
                if changed:
                    self.list_point_name()
                else:
                    QMessageBox.about(self, "Failed", "Fail to edit the name\nname is duplicate.")

            pt_prop = {key: value}

            ## check the name see if it is duplicated
            self.data.set_current_pt_of_current_img_dict(pt_prop)

            self.list_properties()




            # self.data.reset_current_img_point_dict()


    def file_list_current_item_changed(self,row):
        """Detect whether the current image in the file list changed.
        Auto add segmentation classes if quick label mode is enabled
        """
        if row !=-1:
            idx = row

            self.widget_file_list.setCurrentRow(idx)
            img_id = str(self.widget_file_list.currentItem().text())
            print("file_list_current_item_changed:", row,img_id)
            self.data.set_image_id(img_id)

            if self.act_quick_label_mode.isChecked():
                while len(self.data.get_current_image_segments_cv())< len(self.current_quick_segs):
                    name = self.current_quick_segs[len(self.data.get_current_image_segments_cv())]
                    self.data.add_seg_for_current_img(name)



            self.widget_annotation.reset_mask()
            self.widget_annotation.update()


            # # Set table:
            self.list_point_name()
            self.list_properties()

            self.list_seg_name()



    def point_list_current_item_changed(self,row):

        if row !=-1:
            idx_pt = row
            key = self.widget_point_list.item(row).text()
            self.widget_point_list.setCurrentRow(idx_pt)
            self.data.get_current_image().set_current_pt_id(idx_pt)
            self.data.get_current_image().set_current_pt_key(key)

            self.list_properties()

    def segment_list_changed(self, row):
        """
        Set the segmentation colour and the category saved in the datafile.

        :param text:
        :return:
        """

        if row !=-1:
            self.widget_annotation.contour_colour = self.seg_colours[row % len(self.seg_colours)]
            self.widget_annotation.contour_name = self.widget_segment_list.item(row).text()
            self.list_properties()
        else:
            self.widget_annotation.contour_colour = None
            self.widget_annotation.contour_name = None



    def eventFilter(self, source, event):

        modifiers = QApplication.keyboardModifiers()

        if (event.type() == QEvent.Wheel and source is self.scroll_area.viewport()):
            if modifiers and modifiers == Qt.ControlModifier:
                return True

        # Disable right click in file list
        # if (event.type() == QEvent.MouseButtonPress  and \
        #     (source is self.widget_file_list.viewport() or self.widget_point_list.viewport())):
        #     if event.button() == Qt.RightButton:
        #         return True

        # Disable right click in file list
        if event.type() == QEvent.MouseButtonPress  and source is self.widget_file_list.viewport():
            if event.button() == Qt.RightButton:
                return True

        if (event.type() == QEvent.MouseButtonPress or event.type() == QEvent.MouseButtonDblClick)  and self.widget_browser.widget_image_browser.viewport():
            if event.button() == Qt.RightButton:
                return True

        if source == self.widget_annotation and self.data.has_images() ==False:
            return True

        return False


    def toggle_quick_label_mode(self):
        if self.data.has_images():
            if self.act_quick_label_mode.isChecked():
                self.current_quick_points =[item.text() for item in iter_all_list_items(self.widget_point_list)]
                cur_pt_str = "\n".join(["{}.{}".format(idx+1, name) for idx, name in enumerate(self.current_quick_points)])
                self.current_quick_segs = [item.text() for item in iter_all_list_items(self.widget_segment_list)]
                cur_seg_str = "\n".join(["{}.{}".format(idx+1, name) for idx, name in enumerate(self.current_quick_segs)])

                reply = QMessageBox.question(self, "Enable quick label?",
                                             "Points:\n{}\n\nSegmentations:\n{}\n".format(cur_pt_str,cur_seg_str),
                              QMessageBox.Yes |QMessageBox.No)

                if reply == QMessageBox.Yes:
                    ## start the quick label
                    self.widget_quick_label.setVisible(True)
                    # self.info_dock.setVisible(True)
                    self.label_quick_pt.setText("Points:\n"+cur_pt_str)
                    self.label_quick_seg.setText("Segmentations:\n"+cur_seg_str)
                    #  +"\nNote: These segmentation classes are automatically added to images without any segmentaions."
                else:
                    self.widget_quick_label.setVisible(False)
                    # self.info_dock.setVisible(False)
                    self.act_quick_label_mode.setChecked(False)

            else:
                self.widget_quick_label.setVisible(False)
                # self.info_dock.setVisible(False)
        else:
            self.act_quick_label_mode.setChecked(False)
            self.widget_quick_label.setVisible(False)
            # self.info_dock.setVisible(False)

    def toggle_review_mode(self):
        """Toggle the review mode, codes in browser_panel.py
        """        
        
        def toggle_review_mode_UI(on):
            """Control UIs (in the annotation panel) 

            Args:
                on (bool): Whether turn UIs on or off
            """
            self.button_add_seg.setEnabled(on)
            self.button_del_seg.setEnabled(on)
            self.button_del_pt.setEnabled(on)
            for item in iter_all_list_items(self.widget_segment_list):
                if on:
                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                    # item.setFlags(item.flags() | Qt.ItemIsSelectable)
                else:
                    item.setFlags(item.flags() & ~Qt.ItemIsUserCheckable)
                    # item.setFlags(item.flags() | ~Qt.ItemIsSelectable)
                    
        if self.act_browse_mode.isChecked():
            self.widget_stack.setCurrentWidget(self.widget_browser)
            self.widget_browser.prepare_images()
            toggle_review_mode_UI(on = False)
            # self.property_dock.setVisible(False)
        else:
            self.widget_stack.setCurrentWidget(self.scroll_area)
            # self.property_dock.setVisible(True)
            toggle_review_mode_UI(on = True)



    def toggle_flag_img(self):
        """Only show toggle images 
        :return:
        """
        if self.data.has_images():
            flag_mode = self.act_attention_imgs_only.isChecked()
            # Set the data into flag mode
            # By hiding the non-flag images
            flagged_img_idx = self.data.toggle_flag_img(flag_mode)
            # Set hidden item to file list?

            # self.list_file_names()
            print("toggle_flag_img" , flagged_img_idx)
            self.filter_img( flagged_img_idx, flag_mode)

            # self.hide_file_names(flag_mode,flagged_img_idx)
            # self.widget_browser.hide_icons(flag_mode,flagged_img_idx)
            #
            #
            # self.widget_file_list.setCurrentRow(self.data.current_image_id)
            # self.widget_annotation.update()

    def filter_img(self, img_idx, isfilter):

        # self.hide_file_names(isfilter,img_idx)
        self.list_file_names()

        self.widget_annotation.update()



    def hide_file_names(self):
        """Go through self.data
        Check and hide images (non-flag images) that have been filtered 

        Returns:
            bool : Whether has hidden images
        """
        if self.act_attention_imgs_only.isChecked():
            flagged_img_idx = list(set(self.data.filtered_img_idx) & set(self.data.flagged_img_idx))
        else:
            flagged_img_idx = self.data.filtered_img_idx

        first_non_hidden_row = 0
        found_first_non_hidden_row = False
        for idx_row, item in enumerate(iter_all_list_items(self.widget_file_list)):
            if str(item.text()) in flagged_img_idx:
                item.setHidden(False)
                if not found_first_non_hidden_row:
                    first_non_hidden_row = idx_row
                    found_first_non_hidden_row = True
            else:
                item.setHidden(True)

        # Set to the first non-hidden index in the file list.
        self.widget_file_list.setCurrentRow(first_non_hidden_row)

        return found_first_non_hidden_row



    def delete_point(self):
        """
        Action after click delete point
        :return:
        """
        if self.widget_segment_list.currentItem() is not None:
            self.data.remove_pt_for_current_img(self.widget_point_list.currentItem().text())
            self.list_point_name()
            self.list_properties()


    def auto_fill(self):
        """
        Auto fill the current segmentation
        :return:
        """
        try:
            cur_idx = self.widget_segment_list.currentRow()
            cur_color = self.seg_colours[cur_idx%len(self.seg_colours)]

            self.data.close_current_segment(self.widget_annotation.canvas, self.widget_segment_list.currentItem().text(), cur_color)
            self.update_segment_drawing()
        except Exception as e:
            print(e)

    def add_segmentation(self):
        """add the segmentation
        """
        if self.widget_file_list.currentRow()>=0:
            name = self.widget_annotation.get_annotation_name('seg')

            if name:
                if self.data.add_seg_for_current_img(name):
                    print("added")
                else:
                    QMessageBox.about(self, "Failed", "Fail to add the label\nname is duplicate.")

            self.list_seg_name()

    def delete_segmentation(self):
        """Delete the segmentation
        """
        if self.widget_segment_list.currentItem() is not None:
            self.data.remove_seg_for_current_img(self.widget_segment_list.currentItem().text())

            self.list_seg_name()




    def message_unsave(self):
        """
        Message box of saving changes
        :return: save, no cancel,  False for no data.
        """

        if self.data and self.data.changed:
            reply = QMessageBox.question(self, "Saving changes?", "Do you want to save changes to {}".format(self.data.file_name),
                                      QMessageBox.Save |QMessageBox.No| QMessageBox.Cancel)

            if reply == QMessageBox.Save:
                self.save_annotations()

            return reply
        else:
            return False

    def closeEvent(self, event):
        reply = self.message_unsave()
        if reply == QMessageBox.Save or reply == QMessageBox.No:
            event.accept()
        elif reply == QMessageBox.Cancel:
            event.ignore()

    def menu_point_list(self,position):
        """
        Context menu (right click on the point panel) for point list.
        :param position:
        :return:
        """
        if self.widget_point_list.count() != 0:
            menu = QMenu()
            self.act_delete_point.setText("Delete {}".format(self.widget_point_list.currentItem().text()))
            menu.addAction(self.act_delete_point)
            menu.exec_(self.widget_point_list.viewport().mapToGlobal(position))


    def update_file_label(self, changed):
        """
        update app when data changed
        :param changed: Whether the value is change or not
        :return:
        """
        return False
        # if changed == True:
        #     # self.widget_anno_file_label.setText("Annotation file: {}*".format(self.data.file_name))
        #     self.widget_folder_label.setText("Image Dir: {}\nAnnotation file: {}".format(os.path.abspath(self.data.work_dir),os.path.abspath(self.data.file_name)))
        #     self.widget_annotation.update()
        # else:
        #     # self.widget_anno_file_label.setText("Annotation file: {}".format(self.data.file_name))
        #     self.widget_folder_label.setText("Image Dir: {}\nAnnotation file: {}".format(os.path.abspath(self.data.work_dir),os.path.abspath(self.data.file_name)))

    def update_menu_has_imgs(self):
        """
        Enable UI component acts after having images
        :return:
        """

        self.act_sort_file_names.setEnabled(True)
        self.act_sort_anno_names.setEnabled(True)

        self.widget_annotation.act_origin_size.setEnabled(True)
        self.widget_annotation.act_zoom_in.setEnabled(True)
        self.widget_annotation.act_zoom_out.setEnabled(True)

    def update_menu_no_img(self):
        """

        :return:
        """

        return False


    def update_menu_undo(self, changed):
        self.act_undo.setEnabled(changed)


    def sort_file_names(self):
        """
        Sort file names alphabetically

        :return:
        """
        self.data.sort(self.act_sort_file_names.isChecked())
        self.list_file_names()
        # self.widget_file_list.setCurrentRow(0)

    def sort_anno_names(self):
        """
        Sort the annotaion of name alphabetically. Only sort the anno lists
        :return:
        """
        # self.data.set_sort_points(self.act_sort_anno_names.isChecked())

        self.list_point_name()

    def update_segment_drawing(self):
        """
        Update segmentation on the image.
        Paint ticked segmentation

        event: clicking the segmentation panel

        :return:
        """
        self.widget_annotation.reset_mask()
        items = []
        colors = []
        for i_item in range(self.widget_segment_list.count()):
            item = self.widget_segment_list.item(i_item)
            # Draw the certain mask
            if item.checkState() == 2:
                items.append(item)
                colors.append(self.seg_colours[i_item%len(self.seg_colours)])

        self.widget_annotation.draw_init_mask(items, colors)

        self.widget_annotation.update()






if __name__ == '__main__':


    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())
