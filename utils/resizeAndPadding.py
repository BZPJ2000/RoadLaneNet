import os
from PIL import Image, ImageOps


def resizeAndPadding(image, width=800, height=800, save_path="./"):
    # 获取原始图像的长宽
    original_width, original_height = image.size
    min_ori = min(original_width, original_height)
    if min_ori < width:
        width = min_ori
        height = min_ori
    # 计算调整后的图像尺寸
    if original_width >= original_height:
        ratio = float(width) / original_width
        new_width = width
        new_height = int(original_height * ratio)
    else:
        ratio = float(height) / original_height
        new_width = int(original_width * ratio)
        new_height = height

    # 计算需要填充的宽度和高度
    padding_width = width - new_width
    padding_height = height - new_height

    # 计算填充的左边、上边、右边和下边
    if padding_width >= padding_height:
        left = int(padding_width / 2)
        top = 0
    else:
        left = 0
        top = int(padding_height / 2)

    right = padding_width - left
    bottom = padding_height - top

    # 使用resize()调整图像的大小
    image = image.resize((new_width, new_height), Image.ANTIALIAS)

    # 使用ImageOps.pad()函数添加填充
    image = ImageOps.expand(image, border=(left, top, right, bottom), fill=(255, 255, 255))
    image.save(save_path)
    return image


def pic_function(base_path, save_path, width=800, height=800):
    # 打开图像
    image = Image.open(base_path)
    if image.mode == "P" or "RGBA":
        image = image.convert('RGB')

    resizeAndPadding(image=image, save_path=save_path,
                     width=width, height=height)


# 处理一个文件夹中所有图片
def dir_function(base_dir, save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for item in os.listdir(base_dir):
        pic_base_path = base_dir + item
        pic_save_path = save_dir + item
        pic_function(pic_base_path, pic_save_path)


if __name__ == '__main__':
    path1 = './ori2/'
    path2 = './ori/'
    if not os.path.exists(path2):
        os.mkdir(path2)
    for item in os.listdir(path1):
        item_base_path = path1 + item + "/"
        item_save_path = path2 + item + "/"
        dir_function(item_base_path, item_save_path)
