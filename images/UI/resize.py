from PIL import Image
import os


def fixed_size(filePath, savePath, image_width, image_height):
    """按照固定尺寸处理图片"""
    im = Image.open(filePath)
    out = im.resize((image_width, image_height), Image.ANTIALIAS)
    out.save(savePath)


if __name__ == '__main__':
    fixed_size('./a.jpg','./a.jpg', 600, 600)
    fixed_size('./b.jpg', './b.jpg', 600, 600)
    fixed_size('./c.jpg', './c.jpg', 1200, 600)
