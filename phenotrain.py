# from winreg import SetValue
from PyQt5 import QtCore, QtWidgets
# from PyQt5.QtWidgets import QMainWindow, QWidget, QLabel, QLineEdit
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import sys,os
import pandas as pd

# sys.path.append('ui/')
# import deeplearning.ui.train_ui as train_ui
# import deeplearning.ui.pred_ui as pred_ui

from deeplearning.ui import train_ui 
from deeplearning.ui import pred_ui

import deeplearning
from deeplearning import seg_deeplab
from deeplearning import kpt_rcnn
# import deeplearning.deep_lab_seg as deep_lab_seg 
# import deeplearning.kpt_rcnn as kpt_rcnn 
import time

class progress_widget(QWidget):
    def __init__(self):
        super(progress_widget, self).__init__()
        self.setWindowTitle('Progress')
        self.pbar = QProgressBar(self)
        self.pbar.setValue(0)
        self.pbar.setMaximum(10)
        
        self.btn = QPushButton('ok')
        self.btn.clicked.connect(self.btnFunc)
        
        self.label=QLabel()
        
        self.vbox = QVBoxLayout()
        self.vbox.addWidget(self.label)
        self.vbox.addWidget(self.pbar)
        self.vbox.addWidget(self.btn)
        # self.vbox.addWidget(self.btn)
        self.setLayout(self.vbox)
        
        self.resize(500, 200)
        self.setStyleSheet("font: 12pt \"Arial\";")
        # self.show()
    
    def btnFunc(self):
        self.pbar.setValue(0)
        self.close()
        
    def show_app(self):
        self.show()
    
class train_Thread(QThread):
    """The Qthread for training

    Args:
        QThread (_type_): _description_
    """
    _signal = pyqtSignal(str)

    def __init__(self,mode,format, csv_path,img_path,mask_path,scale, lr, batch,num_epochs,test_percent,train_lv,mainWin):
    # def __init__(self):
        super(train_Thread, self).__init__()
        
        self.mode=mode
        self.csv_path = csv_path
        self.img_path=img_path
        self.scale = scale
        self.lr = lr
        self.batch = batch
        self.num_epochs = num_epochs
        self.test_percent = test_percent
        self.train_lv = train_lv
        self.mainWin= mainWin
        self.mask_path=mask_path
        self.format=format
        
    def __del__(self):
        self.wait()

    def run(self):
        
        # try:
        #     kpt_rcnn.train(self.csv_path,self.img_path,self.scale,self.lr,self.batch,
        #                    self.num_epochs,self.test_percent,self.train_lv,self._signal)
        # except Exception as e:        
        #     QMessageBox.warning(self.mainWin, "Warning" , str(e))
        # try:
        if self.mode=="Segmentation":
            if self.format=="CSV":
                seg_deeplab.train(self.img_path, self.scale, self.lr, self.batch,
                            self.num_epochs,self.test_percent,self.train_lv,self._signal, csv_path=self.csv_path,mask_path=None)
            else:
                seg_deeplab.train(self.img_path, self.scale, self.lr, self.batch,
                            self.num_epochs,self.test_percent,self.train_lv,self._signal, csv_path=None,mask_path=self.mask_path)
        elif self.mode=="Point":
            kpt_rcnn.train(self.csv_path,self.img_path,self.scale,self.lr,self.batch,
                        self.num_epochs,self.test_percent,self.train_lv,self._signal)            
        # except Exception as e:        
            # QMessageBox.warning(self.mainWin, "Warning" , "Error message from Python {}\nPlease check the input images and annotations are correct".format(str(e)))
        # for i in range(self.num_epochs):
        #     time.sleep(0.1)
        #     self._signal.emit(str(i))
            
    
class pred_Thread(QThread):
    """The Qthread for training

    Args:
        QThread (_type_): _description_
    """
    _signal = pyqtSignal(str)

    def __init__(self,mode, format,csv_path,img_path,model_path,output_dir, scale,mainWin):
    # def __init__(self):
        super(pred_Thread, self).__init__()
        
        self.mode=mode
        self.format = format
        self.csv_path = csv_path
        self.img_path=img_path
        self.model_path = model_path
        self.output_dir = output_dir
        self.scale = scale

        self.mainWin= mainWin

        
    def __del__(self):
        self.wait()

    def run(self):
        # try:
        if self.mode=="Segmentation":
            seg_deeplab.pred(self.csv_path,self.img_path, self.model_path,self.output_dir,
                        self.format, self.scale,self._signal)

        elif self.mode=="Point":
            kpt_rcnn.pred(self.csv_path,self.img_path, self.model_path,self.output_dir,
                        self.scale,self._signal)
        # except Exception as e:        
        #     QMessageBox.warning(self.mainWin, "Warning" , str(e))
        
        
        # kpt_rcnn.train(csv_path,img_path,scale,lr,batch,num_epochs,train_lv,self._signal)
        # for i in range(self.num_epochs):
        #     time.sleep(0.1)
        #     self._signal.emit(str(i))
            
        # self._signal.emit("traning finishes, the result is saved in:")    


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        # self.widget_dl = tab_ui.Ui_TabWidget()
        self.setWindowTitle('PhenoLearn_DL Toolkit')
        
        self.train_ui = train_ui.Ui_Form()
        self.widget_train = QWidget()
        self.train_ui.setupUi(self.widget_train)
        
        self.pred_ui = pred_ui.Ui_Form()
        self.pred_widget = QWidget()
        self.pred_ui.setupUi(self.pred_widget)
        
        self.widget_tab=QTabWidget()
        self.widget_tab.setStyleSheet("font: 12pt \"Arial\";font-weight: bold;")
        
        self.widget_tab.addTab(self.widget_train, "Train")
        self.widget_tab.addTab(self.pred_widget, "Predict")
        # self.widget_anno_tabs.addTab(self.widget_segment, "Segmentation")
        
        
        
        
        self.train_ui.pushButton_train.clicked.connect(self.train)
        self.train_ui.pushButton_dir.clicked.connect(lambda: self.open_dir("train_img_dir"))
        self.train_ui.pushButton_mask.clicked.connect(lambda: self.open_dir("train_seg_dir"))
        self.train_ui.pushButton_file.clicked.connect(lambda: self.open_file(True))
        
        
        self.pbar_widget = progress_widget()
        
        self.train_ui.lineEdit_lr.setValidator(QDoubleValidator(0.0000000001,1.0,10))
        
        # self.train_ui.lineEdit_train.setValidator(QIntValidator(1,99))
        
        # self.train_ui.lineEdit_train.textEdited.connect(lambda text: self.update_split(text, 'train'))
        self.train_ui.lineEdit_valid.setValidator(QIntValidator(1,99))
        # self.train_ui.lineEdit_valid.textEdited.connect(self.update_lineEdit_valid)
        # self.train_ui.lineEdit_valid.textEdited.connect(lambda text: self.update_split(text, 'valid'))
        
        self.train_ui.comboBox_cate_train.currentIndexChanged.connect(self.train_cate_changed)
        self.train_ui.comboBox_format.currentIndexChanged.connect(self.input_format_changed)
        self.train_ui.pushButton_mask.setEnabled(False)
        
        
        
        self.pred_ui.comboBox_cate.currentIndexChanged.connect(self.pred_cate_changed)
        self.pred_ui.pushButton_predict.clicked.connect(self.pred)
        
        self.pred_ui.pushButton_file.clicked.connect(lambda: self.open_file(False))
        self.pred_ui.pushButton_dir.clicked.connect(lambda: self.open_dir("pred_img_dir"))
        self.pred_ui.pushButton_checkpoint.clicked.connect(self.open_model)
        self.pred_ui.pushButton_output.clicked.connect(self.open_output_dir)

        
        # self.layout = QVBoxLayout(self)
        # self.layout.addWidget(self.widget_tab)
        self.setCentralWidget(self.widget_tab)
        # self.setGeometry(100, 100)
    
    def open_dir(self, mode):
        """Open a directory
        """        
        try:
            # open a dialog about the file
            defaultOpenDirPath = os.path.dirname('.')

            temp = (QFileDialog.getExistingDirectory(self,
                                                            'Open dir for images', defaultOpenDirPath,
                                                            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks))
            if temp:
                # self.work_dir = temp
                if mode=="train_img_dir":    
                    self.train_ui.lineEdit_dir.setText(temp)
                elif mode=="pred_img_dir":
                    self.pred_ui.lineEdit_dir.setText(temp)
                elif mode=="train_seg_dir":
                    self.train_ui.lineEdit_mask.setText(temp)
        except Exception as e:
            print(e)               
                    
    def open_file(self, is_train):
        """Open annotation file
        """
        try:
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getOpenFileName(self, 'Open Annotation File', '.',
                                                  'Files (*.csv)', options=options)
            
            if file_name:    
                # self.anno_file = file_name
                if is_train:  
                    self.train_ui.lineEdit_file.setText(file_name)
                else:
                    self.pred_ui.lineEdit_file.setText(file_name)
        except Exception as e:
            print(e)                
    
    def open_model(self):
        """Open a pth file for reload the model
        """
        try:
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getOpenFileName(self, 'Open Model', '.',
                                                  'Files (*.pth)', options=options)
            
            if file_name:    
                # self.model_file = file_name
                self.pred_ui.lineEdit_checkpoint.setText(file_name)
        except Exception as e:
            print(e)                
    
    
    def open_output_dir(self):
        """Open a directory for the predictions
        """
        try:
            # open a dialog about the file
            # defaultOpenDirPath = os.path.dirname('.')

            temp = (QFileDialog.getExistingDirectory(self,
                                                            'Open a folder for saving the predictions', '.',
                                                            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks))
            if temp:
                # self.output_dir = temp
                self.pred_ui.lineEdit_output.setText(temp)
        except Exception as e:
            print(e)     
    
    def update_split(self, text, input):
        """deprecated
        Update the train/valid split

        Args:
            text (_type_): The text from either training or validation split
            input (_type_): Input flag for train or valid
        """        
        is_valid = self.train_ui.lineEdit_train.hasAcceptableInput() and self.train_ui.lineEdit_valid.hasAcceptableInput()
        # (int(self.train_ui.lineEdit_train.text()) + int(self.train_ui.lineEdit_valid.text()) != 100)
        if text !="" and is_valid:

            if input =="valid":
                self.train_ui.lineEdit_train.setText(str(100 - int(text)))
            elif input == "train":
                self.train_ui.lineEdit_valid.setText(str(100 - int(text)))
        
        else:
            if input =="valid":
                self.train_ui.lineEdit_valid.setText(str(100 - int(self.train_ui.lineEdit_train.text())))
            elif input == "train":
                self.train_ui.lineEdit_train.setText(str(100 - int(self.train_ui.lineEdit_valid.text())))
    

        
    
    def pred_cate_changed(self, row):
        """Change the output format selection box in Predict tab.

        Args:
            row (_type_): Row index of the selected item
        """

        cate = self.pred_ui.comboBox_cate.currentText()
        if cate == "Segmentation":
            # self.pred_ui.comboBox_format.removeItem(1)
            self.pred_ui.comboBox_format.addItem("Mask")
        elif cate=="Point":
            self.pred_ui.comboBox_format.removeItem(1)
            # self.pred_ui.comboBox_format.addItem("JSON")
          
    def train_cate_changed(self, row):
        """Change minimal batch size to 2 if segmentation is selected

        Args:
            row (_type_): Row index of the selected item
        """
        cate = self.train_ui.comboBox_cate_train.currentText()
        if cate == "Segmentation":
            
            self.train_ui.spinBox_batch.setMinimum(2)
            self.train_ui.spinBox_batch.setValue(2)
            self.train_ui.comboBox_format.addItem("Mask")
        elif cate=="Point":
            self.train_ui.spinBox_batch.setMinimum(1)
            self.train_ui.spinBox_batch.setValue(1)
            
            self.train_ui.comboBox_format.removeItem(1)
    
    def input_format_changed(self, row):
        """Update train UI when the input format is changed

        Args:
            row (_type_): Row index of the selected item
        """
        cate = self.train_ui.comboBox_format.currentText()          
        if cate=="CSV":
            self.train_ui.pushButton_file.setEnabled(True)
            
            self.train_ui.pushButton_mask.setEnabled(False)
            self.train_ui.lineEdit_mask.clear()
            
        if cate =="Mask":
            self.train_ui.pushButton_file.setEnabled(False)
            self.train_ui.lineEdit_file.clear()
            
            self.train_ui.pushButton_mask.setEnabled(True)
            
        
    def train(self):
        
        # Check the if the input is correct
        input_check = self.train_ui.lineEdit_valid.hasAcceptableInput() & \
            self.train_ui.lineEdit_lr.hasAcceptableInput() & \
            (self.train_ui.lineEdit_dir.text()!='')
        
        if self.train_ui.comboBox_format.currentText()=="CSV":      
            input_check &= (self.train_ui.lineEdit_file.text()!='')
        if self.train_ui.comboBox_format.currentText()=="Mask": 
            input_check &= (self.train_ui.lineEdit_mask.text()!='')
        
        train_lv_str = self.train_ui.comboBox_train_lv.currentText()
        
        if train_lv_str=="Minimal":
            train_lv="1"
        elif train_lv_str=="Intermediate":
            train_lv="2"
        elif train_lv_str=="Full":
            train_lv="3"
        
        if not input_check:
             QMessageBox.about(self, "Information", "Please check if all settings are correct.")
        else:
            # Take all the configurations from UI
            num_epochs = self.train_ui.spinBox_epoch.value()
            #The self.train_ui.spinBox_scale.value() is the image resize percentage
            scale = 1/(self.train_ui.spinBox_scale.value()/100)
            batch = self.train_ui.spinBox_batch.value()
            
            mode = self.train_ui.comboBox_cate_train.currentText()
            
            
            test_percent = int(self.train_ui.lineEdit_valid.text())
            
            format=self.train_ui.comboBox_format.currentText()
            mask_path=self.train_ui.lineEdit_mask.text()
            


            # Check if the csv and folder are good
            
            lr = float(self.train_ui.lineEdit_lr.text())
            
            # Set the maximum and text of the progress bar
            self.pbar_widget.pbar.setFormat("Epochs: %v/{}".format(num_epochs))
            self.pbar_widget.pbar.setMaximum(num_epochs)
            
            # print(epoch,lr)
            csv_path=self.train_ui.lineEdit_file.text()
            img_path=self.train_ui.lineEdit_dir.text()
            
                
            
            self.train_thread = train_Thread(mode=mode,format=format ,
                                            csv_path = csv_path,img_path=img_path,mask_path=mask_path,
                                        scale=scale, lr =lr, batch = batch,
                                        num_epochs=num_epochs,test_percent=test_percent,train_lv =train_lv,mainWin=self )
            
            # self.train_thread = train_Thread(mode=mode , csv_path = "",
            #                         img_path="self.work_dir",scale=scale, lr =lr, batch = batch,
            #                         num_epochs=num_epochs,test_percent=test_percent,train_lv =train_lv,mainWin=self )
            
            self.train_thread._signal.connect(self.signal_accept)
            self.train_thread.start()
            
            # Set main window unable
            self.setEnabled(False)
            # Set progress bar window
            self.pbar_widget.show()
            self.pbar_widget.btn.setEnabled(False)
            self.pbar_widget.label.setText("Training...")
        
        
        
    def pred(self):
        mode = self.pred_ui.comboBox_cate.currentText()
        format = self.pred_ui.comboBox_format.currentText()
        #The self.pred_ui.spinBox_scale.value() is the image resize percentage
        scale = 1/(self.pred_ui.spinBox_scale.value()/100)
        
        csv_path = self.pred_ui.lineEdit_file.text()
        img_path = self.pred_ui.lineEdit_dir.text()
        model_path = self.pred_ui.lineEdit_checkpoint.text()
        output_dir = self.pred_ui.lineEdit_output.text()
        
        
        # Set the maximum and text of the progress bar
        df_temp=pd.read_csv(csv_path)
        num_img = df_temp.shape[0]    
        self.pbar_widget.pbar.setFormat("Images: %v/{}".format(num_img))
        self.pbar_widget.pbar.setMaximum(num_img)
        
        self.pred_thread = pred_Thread(mode=mode,format=format,
                                       csv_path=csv_path,img_path=img_path,
                                       model_path=model_path,output_dir=output_dir, 
                                       scale=scale,mainWin = self)
        self.pred_thread._signal.connect(self.signal_accept)
        self.pred_thread.start()
        
        # Set main window unable
        self.setEnabled(False)
        # Set progress bar window
        self.pbar_widget.show()
        self.pbar_widget.btn.setEnabled(False)
        
        
    def signal_accept(self, msg):
        # Get the signal from training thread
        if isinstance(msg, int) or msg.isnumeric():
            # update progress bar
            self.pbar_widget.pbar.setValue(int(msg)+1)
            
            # When finishes, enable ok button and the app.
            if int(msg)+1 == self.pbar_widget.pbar.maximum():
                self.setEnabled(True)
                self.pbar_widget.btn.setEnabled(True)
        else:
            # Show training detail on the window.
            
            self.pbar_widget.label.setText(msg)
      
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())