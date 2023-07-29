# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'pred_ui.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1398, 192)
        Form.setStyleSheet("font: 12pt \"Arial\";")
        self.gridLayout_2 = QtWidgets.QGridLayout(Form)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.lineEdit_output = QtWidgets.QLineEdit(Form)
        self.lineEdit_output.setText("")
        self.lineEdit_output.setReadOnly(True)
        self.lineEdit_output.setObjectName("lineEdit_output")
        self.gridLayout.addWidget(self.lineEdit_output, 2, 3, 1, 1)
        self.spinBox_scale = QtWidgets.QSpinBox(Form)
        self.spinBox_scale.setMinimum(1)
        self.spinBox_scale.setMaximum(100)
        self.spinBox_scale.setProperty("value", 100)
        self.spinBox_scale.setObjectName("spinBox_scale")
        self.gridLayout.addWidget(self.spinBox_scale, 2, 5, 1, 1)
        self.pushButton_output = QtWidgets.QPushButton(Form)
        self.pushButton_output.setObjectName("pushButton_output")
        self.gridLayout.addWidget(self.pushButton_output, 2, 2, 1, 1)
        self.label = QtWidgets.QLabel(Form)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 2, 4, 1, 1)
        self.label_6 = QtWidgets.QLabel(Form)
        self.label_6.setObjectName("label_6")
        self.gridLayout.addWidget(self.label_6, 0, 0, 1, 1)
        self.pushButton_checkpoint = QtWidgets.QPushButton(Form)
        self.pushButton_checkpoint.setObjectName("pushButton_checkpoint")
        self.gridLayout.addWidget(self.pushButton_checkpoint, 0, 4, 1, 1)
        self.comboBox_format = QtWidgets.QComboBox(Form)
        self.comboBox_format.setObjectName("comboBox_format")
        self.comboBox_format.addItem("")
        self.gridLayout.addWidget(self.comboBox_format, 0, 3, 1, 1)
        self.pushButton_dir = QtWidgets.QPushButton(Form)
        self.pushButton_dir.setObjectName("pushButton_dir")
        self.gridLayout.addWidget(self.pushButton_dir, 0, 6, 1, 1)
        self.lineEdit_checkpoint = QtWidgets.QLineEdit(Form)
        self.lineEdit_checkpoint.setText("")
        self.lineEdit_checkpoint.setReadOnly(True)
        self.lineEdit_checkpoint.setObjectName("lineEdit_checkpoint")
        self.gridLayout.addWidget(self.lineEdit_checkpoint, 0, 5, 1, 1)
        self.comboBox_cate = QtWidgets.QComboBox(Form)
        self.comboBox_cate.setObjectName("comboBox_cate")
        self.comboBox_cate.addItem("")
        self.comboBox_cate.addItem("")
        self.gridLayout.addWidget(self.comboBox_cate, 0, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 0, 2, 1, 1)
        self.pushButton_file = QtWidgets.QPushButton(Form)
        self.pushButton_file.setObjectName("pushButton_file")
        self.gridLayout.addWidget(self.pushButton_file, 2, 0, 1, 1)
        self.lineEdit_file = QtWidgets.QLineEdit(Form)
        self.lineEdit_file.setReadOnly(True)
        self.lineEdit_file.setObjectName("lineEdit_file")
        self.gridLayout.addWidget(self.lineEdit_file, 2, 1, 1, 1)
        self.lineEdit_dir = QtWidgets.QLineEdit(Form)
        self.lineEdit_dir.setReadOnly(True)
        self.lineEdit_dir.setObjectName("lineEdit_dir")
        self.gridLayout.addWidget(self.lineEdit_dir, 0, 8, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)
        self.pushButton_predict = QtWidgets.QPushButton(Form)
        self.pushButton_predict.setObjectName("pushButton_predict")
        self.gridLayout_2.addWidget(self.pushButton_predict, 1, 0, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.pushButton_output.setText(_translate("Form", "Choose the output folder"))
        self.label.setText(_translate("Form", "Image Resize Percentage"))
        self.label_6.setText(_translate("Form", "Model Type"))
        self.pushButton_checkpoint.setText(_translate("Form", "Choose model"))
        self.comboBox_format.setItemText(0, _translate("Form", "CSV"))
        self.pushButton_dir.setText(_translate("Form", "Image Folder"))
        self.comboBox_cate.setItemText(0, _translate("Form", "Point"))
        self.comboBox_cate.setItemText(1, _translate("Form", "Segmentation"))
        self.label_2.setText(_translate("Form", "Output Format"))
        self.pushButton_file.setText(_translate("Form", "Image Name File"))
        self.pushButton_predict.setText(_translate("Form", "Predict"))
