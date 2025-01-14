# -*- coding: utf-8 -*-
# 图片不能在中文路径下，否则程序会崩溃
import shutil
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QLabel, QComboBox
import threading
import os
import sys
from pathlib import Path
import cv2
import os.path as osp
from unet import Unet
from utils.resizeAndPadding import pic_function, resizeAndPadding
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

# -------------------------------------------------------------------#
#   获取当前文件的路径和根目录
# -------------------------------------------------------------------#
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


# -------------------------------------------------------------------#
#   窗口主类
# -------------------------------------------------------------------#
class MainWindow(QTabWidget):
    def __init__(self):
        # 初始化界面
        super().__init__()
        self.setWindowTitle('UNet车道线图像视频语义分割系统')
        self.resize(1200, 800)
        # 图片读取进程
        self.output_size = 600
        self.img2predict = ""
        self.video_path = 0
        self.model_path = './MODEL/line_best_epoch_weights.pth'
        self.mix_type = 0  # 是否让识别结果和原图混合 0；混合 1：不混合
        self.backbone = 'vgg'
        # self.num_classes = 4
        self.num_classes = 2
        self.cuda = False
        self.input_shape = [512, 512]
        self.model = Unet(model_path=self.model_path,
                          mix_type=self.mix_type,
                          backbone=self.backbone,
                          num_classes=self.num_classes,
                          cuda=self.cuda,
                          input_shape=self.input_shape)
        self.vid_source = 0  # 初始设置为摄像头
        self.webcam = True
        self.stopEvent = threading.Event()
        self.stopEvent.clear()
        self.initUI()

    # -------------------------------------------------------------------#
    #   界面初始化
    # -------------------------------------------------------------------#
    def initUI(self):
        # 图片检测子界面
        font_title = QFont('楷体', 16)
        font_main = QFont('楷体', 14)
        # 图片识别界面, 两个按钮，上传图片和显示结果
        img_detection_widget = QWidget()
        img_detection_layout = QVBoxLayout()
        # img_detection_title = QLabel("图片识别功能")
        # img_detection_title.setFont(font_title)
        mid_img_widget = QWidget()
        mid_img_layout = QHBoxLayout()
        self.left_img = QLabel()
        self.right_img = QLabel()
        self.left_img.setPixmap(QPixmap("images/UI/a.jpg"))
        self.right_img.setPixmap(QPixmap("images/UI/b.jpg"))
        self.left_img.setAlignment(Qt.AlignCenter)
        self.right_img.setAlignment(Qt.AlignCenter)
        mid_img_layout.addWidget(self.left_img)
        mid_img_layout.addStretch(0)
        mid_img_layout.addWidget(self.right_img)
        mid_img_widget.setLayout(mid_img_layout)
        up_img_button = QPushButton("图片上传")
        det_img_button = QPushButton("开始分割")

        self.det_label = QLabel('模型选择')
        self.det_com_box = QComboBox()
        self.det_com_box.addItem('模型1：基础模型（仅检测实线车道线，题目需求）')  # 下拉框添加选项，添加单个选项
        self.det_com_box.addItems(['模型2：扩展功能，可检测实现车道线和虚线车道线', '模型3：扩展功能，可检测实线车道线、虚线车道线、人行道','模型4：实线车道线补全'])  # 批量添加选项，支持列表方式
        img_detection_layout.addWidget(self.det_label)
        img_detection_layout.addWidget(self.det_com_box)

        self.det_com_box.currentIndexChanged.connect(self.check_box)  # 绑定索引变化事件

        up_img_button.clicked.connect(self.upload_img)
        det_img_button.clicked.connect(self.seg_img)
        up_img_button.setFont(font_main)
        det_img_button.setFont(font_main)
        up_img_button.setStyleSheet("QPushButton{color:white}"
                                    "QPushButton:hover{background-color: rgb(2,110,180);}"
                                    "QPushButton{background-color:rgb(128,0,128)}"    #(48,124,208)
                                    "QPushButton{border:2px}"
                                    "QPushButton{border-radius:5px}"
                                    "QPushButton{padding:5px 5px}"
                                    "QPushButton{margin:5px 5px}")
        det_img_button.setStyleSheet("QPushButton{color:white}"
                                     "QPushButton:hover{background-color: rgb(2,110,180);}"
                                     "QPushButton{background-color:rgb(128,0,128)}"
                                     "QPushButton{border:2px}"
                                     "QPushButton{border-radius:5px}"
                                     "QPushButton{padding:5px 5px}"
                                     "QPushButton{margin:5px 5px}")
        # img_detection_layout.addWidget(img_detection_title, alignment=Qt.AlignCenter)
        img_detection_layout.addWidget(mid_img_widget, alignment=Qt.AlignCenter)
        img_detection_layout.addWidget(up_img_button)
        img_detection_layout.addWidget(det_img_button)
        img_detection_widget.setLayout(img_detection_layout)

        # todo 视频识别界面
        # 视频识别界面的逻辑比较简单，基本就从上到下的逻辑
        vid_detection_widget = QWidget()
        vid_detection_layout = QVBoxLayout()
        self.det_label1 = QLabel('模型选择')
        self.det_com_box1 = QComboBox()
        self.det_com_box1.addItem('模型1：基础模型（仅检测实线车道线，题目需求）')  # 下拉框添加选项，添加单个选项
        self.det_com_box1.addItems(
            ['模型2：功能扩展，可检测实现车道线和虚线车道线', '模型3：功能扩展，可检测实线车道线、虚线车道线、人行道','模型4：实线车道线补全'])  # 批量添加选项，支持列表方式
        vid_detection_layout.addWidget(self.det_label1)
        vid_detection_layout.addWidget(self.det_com_box1)
        #下拉框消息相应函数
        self.det_com_box1.currentIndexChanged.connect(self.check_box1)  # 绑定索引变化事件

        self.vid_img = QLabel()
        self.vid_img.setPixmap(QPixmap("images/UI/c.jpg"))
        # vid_title.setAlignment(Qt.AlignCenter)
        self.vid_img.setAlignment(Qt.AlignCenter)
        self.webcam_detection_btn = QPushButton("摄像头实时分割（需要外接usb摄像头）")
        self.mp4_detection_btn = QPushButton("视频文件分割")
        self.vid_stop_btn = QPushButton("停止分割")
        self.webcam_detection_btn.setFont(font_main)
        self.mp4_detection_btn.setFont(font_main)
        self.vid_stop_btn.setFont(font_main)
        self.webcam_detection_btn.setStyleSheet("QPushButton{color:white}"
                                                "QPushButton:hover{background-color: rgb(2,110,180);}"
                                                "QPushButton{background-color:rgb(128,0,128)}"
                                                "QPushButton{border:2px}"
                                                "QPushButton{border-radius:5px}"
                                                "QPushButton{padding:5px 5px}"
                                                "QPushButton{margin:5px 5px}")
        self.mp4_detection_btn.setStyleSheet("QPushButton{color:white}"
                                             "QPushButton:hover{background-color: rgb(2,110,180);}"
                                             "QPushButton{background-color:rgb(128,0,128)}"
                                             "QPushButton{border:2px}"
                                             "QPushButton{border-radius:5px}"
                                             "QPushButton{padding:5px 5px}"
                                             "QPushButton{margin:5px 5px}")
        self.vid_stop_btn.setStyleSheet("QPushButton{color:white}"
                                        "QPushButton:hover{background-color: rgb(2,110,180);}"
                                        "QPushButton{background-color:rgb(128,0,128)}"
                                        "QPushButton{border:2px}"
                                        "QPushButton{border-radius:5px}"
                                        "QPushButton{padding:5px 5px}"
                                        "QPushButton{margin:5px 5px}")
        self.webcam_detection_btn.clicked.connect(self.open_cam)
        self.mp4_detection_btn.clicked.connect(self.open_mp4)
        self.vid_stop_btn.clicked.connect(self.close_vid)
        # 添加组件到布局上
        # vid_detection_layout.addWidget(vid_title)
        vid_detection_layout.addWidget(self.vid_img)
        vid_detection_layout.addWidget(self.webcam_detection_btn)
        vid_detection_layout.addWidget(self.mp4_detection_btn)
        vid_detection_layout.addWidget(self.vid_stop_btn)
        vid_detection_widget.setLayout(vid_detection_layout)

        self.left_img.setAlignment(Qt.AlignCenter)
        self.addTab(img_detection_widget, '图片分割')
        self.addTab(vid_detection_widget, '视频分割')

    # -------------------------------------------------------------------#
    #   下拉复选框消息相应函数
    # -------------------------------------------------------------------#
    def check_box(self,i):
        if i == 0:
            self.model_path = './MODEL/line_best_epoch_weights.pth'
            self.model = Unet(model_path=self.model_path,
                              mix_type=self.mix_type,
                              backbone=self.backbone,
                              num_classes=2,
                              cuda=self.cuda,
                              input_shape=self.input_shape)
        elif i == 1:
            self.model_path = './MODEL/3_best_epoch_weights.pth'
            self.model = Unet(model_path=self.model_path,
                              mix_type=self.mix_type,
                              backbone=self.backbone,
                              num_classes=3,
                              cuda=self.cuda,
                              input_shape=self.input_shape)
        elif i == 2:
            self.model_path = './MODEL/4_best_epoch_weights.pth'
            self.model = Unet(model_path=self.model_path,
                              mix_type=self.mix_type,
                              backbone=self.backbone,
                              num_classes=4,
                              cuda=self.cuda,
                              input_shape=self.input_shape)
        elif i == 3:
            self.model_path = './MODEL/full_best_epoch_weights.pth'
            self.model = Unet(model_path=self.model_path,
                              mix_type=self.mix_type,
                              backbone=self.backbone,
                              num_classes=2,
                              cuda=self.cuda,
                              input_shape=self.input_shape)
        else:
            pass

    # -------------------------------------------------------------------#
    #   下拉复选框消息相应函数
    # -------------------------------------------------------------------#
    def check_box1(self, i):
        if i == 0:
            self.model_path = './MODEL/line_best_epoch_weights.pth'
            self.model = Unet(model_path=self.model_path,
                              mix_type=self.mix_type,
                              backbone=self.backbone,
                              num_classes=2,
                              cuda=self.cuda,
                              input_shape=self.input_shape)
        elif i == 1:
            self.model_path = './MODEL/3_best_epoch_weights.pth'
            self.model = Unet(model_path=self.model_path,
                              mix_type=self.mix_type,
                              backbone=self.backbone,
                              num_classes=3,
                              cuda=self.cuda,
                              input_shape=self.input_shape)
        elif i == 2:
            self.model_path = './MODEL/4_best_epoch_weights.pth'
            self.model = Unet(model_path=self.model_path,
                              mix_type=self.mix_type,
                              backbone=self.backbone,
                              num_classes=4,
                              cuda=self.cuda,
                             input_shape=self.input_shape)
        elif i == 3:
            self.model_path = './MODEL/full_best_epoch_weights.pth'
            self.model = Unet(model_path=self.model_path,
                              mix_type=self.mix_type,
                              backbone=self.backbone,
                              num_classes=2,
                              cuda=self.cuda,
                              input_shape=self.input_shape)
        else:
            pass

    # -------------------------------------------------------------------#
    #   上传图片
    # -------------------------------------------------------------------#
    def upload_img(self):
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.jpg *.png *.tif *.jpeg')
        if fileName:
            suffix = fileName.split(".")[-1]
            save_path = osp.join("images/tmp", "tmp_upload." + suffix)
            shutil.copy(fileName, save_path)
            # 应该调整一下图片的大小，然后统一防在一起
            im0 = cv2.imread(save_path)
            resize_scale = self.output_size / im0.shape[0]
            im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
            cv2.imwrite("images/tmp/upload_show_result2.jpg", im0)
            pic_function("images/tmp/upload_show_result2.jpg", "images/tmp/upload_show_result.jpg",
                         self.output_size, self.output_size)
            self.img2predict = fileName
            self.left_img.setPixmap(QPixmap("images/tmp/upload_show_result.jpg"))
            # todo 上传图片之后右侧的图片重置，
            self.right_img.setPixmap(QPixmap("images/UI/c.jpg"))

    # -------------------------------------------------------------------#
    #   分割图片
    # -------------------------------------------------------------------#
    def seg_img(self):
        output_size = self.output_size
        source = self.img2predict  # file/dir/URL/glob, 0 for webcam
        print(source)
        if source == "":
            QMessageBox.warning(self, "请上传", "请先上传图片再进行分割")
        else:
            output = 'images/tmp/single_result.jpg'
            source = str(source)
            image = Image.open(source)
            r_image = self.model.detect_image(image)
            resizeAndPadding(r_image, width=self.output_size, height=self.output_size, save_path=output)
            self.right_img.setPixmap(QPixmap(output))

    # -------------------------------------------------------------------#
    #   视频关闭事件
    # -------------------------------------------------------------------#
    def open_cam(self):
        self.webcam_detection_btn.setEnabled(False)
        self.mp4_detection_btn.setEnabled(False)
        self.vid_stop_btn.setEnabled(True)
        self.vid_source = 0
        self.webcam = True
        # 把按钮给他重置了
        # print("GOGOGO")
        th = threading.Thread(target=self.seg_video)
        th.start()

    # -------------------------------------------------------------------#
    #   开启视频文件检测事件
    # -------------------------------------------------------------------#
    def open_mp4(self):
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.mp4 *.avi')
        self.webcam = False
        if fileName:
            self.webcam_detection_btn.setEnabled(False)
            self.mp4_detection_btn.setEnabled(False)
            # self.vid_stop_btn.setEnabled(True)
            self.vid_source = fileName
            self.webcam = False
            th = threading.Thread(target=self.seg_video)
            th.start()

    # -------------------------------------------------------------------#
    #   视频开启事件
    # -------------------------------------------------------------------#
    def seg_video(self):
        # 打开视频文件
        cap = cv2.VideoCapture(self.vid_source)
        # 循环便利每一帧
        while cap.isOpened():
            # 读取一帧
            ret, frame = cap.read()
            if ret:
                # 处理这一帧
                out_put = "images/tmp/single_result_vid.jpg"
                img_Image = Image.fromarray(np.uint8(frame))
                r_image = self.model.detect_image(img_Image, count=False, name_classes=None)
                r_image.save(out_put)
                self.vid_img.setPixmap(QPixmap(out_put))
            else:
                # 读取完毕，退出循环
                break
            if cv2.waitKey(25) & self.stopEvent.is_set() == True:
                self.stopEvent.clear()
                self.webcam_detection_btn.setEnabled(True)
                self.mp4_detection_btn.setEnabled(True)
                self.reset_vid()
                break
        # 释放资源
        self.reset_vid()
        cap.release()

    # -------------------------------------------------------------------#
    #   上传视频
    # -------------------------------------------------------------------#
    def upload_video(self):
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.mp4 *.wmv')
        if fileName:
            self.video_path = fileName
            # print(self.video_path)

    # -------------------------------------------------------------------#
    #   分割视频
    # -------------------------------------------------------------------#
    def detect_video(self):
        predict_vedio(self.video_path, "", self.model)

    # -------------------------------------------------------------------#
    #   分割摄像头
    # -------------------------------------------------------------------#
    def detect_camera(self):
        predict_vedio(0, "", self.model)

    # -------------------------------------------------------------------#
    #   关闭事件
    # -------------------------------------------------------------------#
    def closeEvent(self, event):
        reply = QMessageBox.question(self,
                                     '提示',
                                     "确定退出吗?",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.close()
            event.accept()
        else:
            event.ignore()

    # -------------------------------------------------------------------#
    #   界面重置事件
    # -------------------------------------------------------------------#
    def reset_vid(self):
        self.webcam_detection_btn.setEnabled(True)
        self.mp4_detection_btn.setEnabled(True)
        self.vid_img.setPixmap(QPixmap("images/UI/c.jpg"))
        self.vid_source = 0
        self.webcam = True

    # -------------------------------------------------------------------#
    #   视频重置事件
    # -------------------------------------------------------------------#
    def close_vid(self):
        self.stopEvent.set()
        self.reset_vid()


# -------------------------------------------------------------------#
#   主函数
# -------------------------------------------------------------------#
if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())