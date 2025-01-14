import os
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import transforms
import time
import cv2
import numpy as np
from PIL import Image
from unet import Unet


# -------------------------------------------------------------------#
#   预测单张图像
# -------------------------------------------------------------------#
def predict(img_path, out_put, model):
    """
    对单张图像进行预测并保存结果。
    :param img_path: 输入图像路径
    :param out_put: 输出图像保存路径
    :param model: 使用的模型
    """
    # 加载模型
    model = Unet(model_path=weights_path, num_classes=2)
    try:
        # 打开图像
        image = Image.open(img_path)
    except:
        print('Open Error! Try again!')
    else:
        # 对图像进行预测
        r_image = model.detect_image(image)
        # 保存预测结果
        r_image.save(out_put)


# -------------------------------------------------------------------#
#   预测视频
# -------------------------------------------------------------------#
def predict_vedio(video_path, video_save_path, unet):
    """
    对视频进行逐帧预测并保存结果。
    :param video_path: 输入视频路径
    :param video_save_path: 输出视频保存路径
    :param unet: 使用的模型
    """
    # 加载模型
    unet = Unet(model_path=weights_path)
    # 打开视频
    capture = cv2.VideoCapture(video_path)
    if video_save_path != "":
        # 设置视频编码器
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # 获取视频帧的宽度和高度
        size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        # 创建视频写入对象
        out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

    # 读取第一帧
    ref, frame = capture.read()
    if not ref:
        raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")

    # 初始化帧率计算
    fps = 0.0
    while (True):
        t1 = time.time()
        # 读取某一帧
        ref, frame = capture.read()
        if not ref:
            break
        # 格式转变，BGRtoRGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 转变成Image
        frame = Image.fromarray(np.uint8(frame))
        # 进行检测
        frame = np.array(unet.detect_image(frame))
        # RGBtoBGR满足opencv显示格式
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 计算帧率
        fps = (fps + (1. / (time.time() - t1))) / 2
        print("fps= %.2f" % (fps))
        # 在帧上显示帧率
        frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 显示帧
        cv2.imshow("video", frame)
        c = cv2.waitKey(1) & 0xff
        if video_save_path != "":
            # 写入帧到输出视频
            out.write(frame)

        # 按下ESC键退出
        if c == 27:
            capture.release()
            break
    print("Video Detection Done!")
    capture.release()
    if video_save_path != "":
        print("Save processed video to the path :" + video_save_path)
        out.release()
    cv2.destroyAllWindows()


# -------------------------------------------------------------------#
#   主函数
# -------------------------------------------------------------------#
if __name__ == '__main__':
    # 模型权重路径
    weights_path = "./MODEL/full_best_epoch_weights.pth"
    # 输出图像路径
    output = './test_pic/result-170927_064343490_Camera_6.png'
    # 输入图像路径
    img_path = "./test_pic/170927_064343490_Camera_6.jpg"
    # 调用预测函数
    predict(img_path, output, weights_path)