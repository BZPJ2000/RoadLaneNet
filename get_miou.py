import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from unet import Unet
from utils.utils_metrics import compute_mIoU, show_results

if __name__ == "__main__":
    # -------------------------------------------------------------------#
    #   model_path 指向 logs 文件夹下的权值文件
    #   训练好后 logs 文件夹下存在多个权值文件，选择验证集损失较低的即可。
    #   验证集损失较低不代表 miou 较高，仅代表该权值在验证集上泛化性能较好。
    # -------------------------------------------------------------------#
    model_path = './MODEL/full_best_epoch_weights.pth'

    # --------------------------------#
    #   所使用的的主干网络：vgg、resnet50
    # --------------------------------#
    backbone = 'vgg'

    # ------------------------------#
    #   分类个数 +1，如 2+1
    # ------------------------------#
    num_classes = 2

    # ---------------------------------------------------------------------------#
    #   miou_mode 用于指定该文件运行时计算的内容
    #   miou_mode 为 0 代表整个 miou 计算流程，包括获得预测结果、计算 miou。
    #   miou_mode 为 1 代表仅仅获得预测结果。
    #   miou_mode 为 2 代表仅仅计算 miou。
    # ---------------------------------------------------------------------------#
    miou_mode = 0

    # --------------------------------------------#
    #   区分的种类，和 json_to_dataset 里面的一样
    # --------------------------------------------#
    name_classes = ["_background_", "SolidLine"]

    # -------------------------------------------------------#
    #   指向 VOC 数据集所在的文件夹
    #   默认指向根目录下的 VOC 数据集
    # -------------------------------------------------------#
    VOCdevkit_path = './game'

    # --------------------------------#
    #   输入图片的大小
    # --------------------------------#
    input_shape = [512, 512]

    # -------------------------------------------------#
    #   mix_type 参数用于控制检测结果的可视化方式
    #
    #   mix_type = 0 的时候代表原图与生成的图进行混合
    #   mix_type = 1 的时候代表仅保留生成的图
    #   mix_type = 2 的时候代表仅扣去背景，仅保留原图中的目标
    # -------------------------------------------------#
    mix_type = 0

    # --------------------------------#
    #   是否使用 Cuda
    #   没有 GPU 可以设置成 False
    # --------------------------------#
    cuda = False

    ######################################################################################################################
    # 清空目录中的所有内容
    def clear_directory_contents(directory):
        """
        清空指定目录中的所有文件和子目录。
        :param directory: 需要清空的目录路径
        """
        # 确保目录存在
        if not os.path.exists(directory):
            print(f"The directory '{directory}' does not exist.")
            return

        # 列出目录下的所有条目
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            # 如果条目是文件，直接删除
            if os.path.isfile(item_path):
                os.remove(item_path)
            # 如果条目是目录，递归调用自身来清空子目录
            elif os.path.isdir(item_path):
                clear_directory_contents(item_path)
                os.rmdir(item_path)

    # 在写入结果到该目录之前，先清空该目录
    print("正在清空目标路径......")
    clear_directory_contents("./miou_out/detection-results")
    print("清空目标路径完成！！！")
    ######################################################################################################################

    # -------------------------------------------------------------------#
    #   加载验证集图像 ID
    # -------------------------------------------------------------------#
    image_ids = open(os.path.join(VOCdevkit_path, "game_data_AB/ImageSets/Segmentation/val.txt"), 'r').read().splitlines()

    # -------------------------------------------------------------------#
    #   设置 ground truth 目录和预测结果保存目录
    # -------------------------------------------------------------------#
    gt_dir = os.path.join(VOCdevkit_path, "game_data_AB/SegmentationClass/")
    miou_out_path = "miou_out"
    pred_dir = os.path.join(miou_out_path, 'detection-results')

    # -------------------------------------------------------------------#
    #   如果 miou_mode 为 0 或 1，生成预测结果
    # -------------------------------------------------------------------#
    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        print("Load model.")
        # 加载 U-Net 模型
        unet = Unet(model_path=model_path, backbone=backbone, mix_type=mix_type,
                    input_shape=input_shape, num_classes=num_classes, cuda=cuda)
        print("Load model done.")

        print("Get predict result.")
        # 遍历验证集图像，生成预测结果
        for image_id in tqdm(image_ids):
            image_path = os.path.join(VOCdevkit_path, "game_data_AB/JPEGImages/" + image_id + ".jpg")
            image = Image.open(image_path)
            image = unet.get_miou_png(image)
            image.save(os.path.join(pred_dir, image_id + ".png"))
        print("Get predict result done.")

    # -------------------------------------------------------------------#
    #   如果 miou_mode 为 0 或 2，计算 mIoU
    # -------------------------------------------------------------------#
    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        # 计算 mIoU
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, name_classes)
        print("Get miou done.")
        # 显示结果
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)


# -------------------------------------------------------------------#
#   反向转化函数：将图像像素值从 1 转换为 255
# -------------------------------------------------------------------#
def modify_and_save_image_new(img_array, save_directory, image_name):
    """
    修改图像像素值并保存图像。
    :param img_array: 图像数组
    :param save_directory: 保存目录
    :param image_name: 图像名称
    :return: 保存路径
    """
    # 修改图像像素：将像素值为 1 的像素设置为 255
    modified_array = np.where(img_array == 1, 255, img_array)
    modified_img = Image.fromarray(modified_array.astype('uint8'), 'L')

    # 保存图像到指定目录
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    save_path = os.path.join(save_directory, image_name)
    modified_img.save(save_path)
    return save_path


# -------------------------------------------------------------------#
#   指定图像所在的目录和保存图像的目录
# -------------------------------------------------------------------#
directory = "./miou_out/detection-results"
save_directory = "./miou_out/detection-results"


# -------------------------------------------------------------------#
#   从目录加载所有图像文件
# -------------------------------------------------------------------#
def load_images_from_directory(directory):
    """
    从给定目录加载图像文件列表。
    :param directory: 图像目录
    :return: 图像文件路径列表
    """
    images = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            images.append(os.path.join(directory, filename))
    return images


# -------------------------------------------------------------------#
#   分析图像像素分布
# -------------------------------------------------------------------#
def analyze_image_pixels(image_path):
    """
    分析图像的像素分布。
    :param image_path: 图像路径
    :return: 像素分布字典和图像数组
    """
    # 打开图像并转换为灰度模式
    img = Image.open(image_path).convert('L')  # 'L' 模式是灰度模式
    img_array = np.array(img)

    # 获取并返回像素值的分布
    unique_pixels, counts = np.unique(img_array, return_counts=True)
    pixel_distribution = dict(zip(unique_pixels, counts))
    return pixel_distribution, img_array


# -------------------------------------------------------------------#
#   从目录加载所有图像文件并处理
# -------------------------------------------------------------------#
print("正在保存结果图片到 './miou_out/detection-results'...")
image_paths = load_images_from_directory(directory)

if image_paths:
    for image_path in image_paths:
        # 分析并打印像素分布
        pixel_distribution, img_array = analyze_image_pixels(image_path)

        # 修改图像并保存
        image_name = os.path.basename(image_path)
        modified_image_path = modify_and_save_image_new(img_array, save_directory, image_name)
else:
    print("目录中未找到图像文件。")
print("结果保存成功！！！")