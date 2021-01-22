import requests
import re
import os
from PIL import Image
from io import BytesIO


'''重命名文件,filePath为文件夹路径，pre_name为自定义的修改后的前缀名，renum为自定义的起始数，fileType为自定义的文件后缀名'''
def rename(filePath,pre_name,renum):
    path = filePath
    name = pre_name
    startNumber = str(renum)
    print("正在将文件重命名为"+name+startNumber+"形式的文件名")
    count=0
    filelist=os.listdir(path)
    for files in filelist:
        Olddir = os.path.join(path, files)
        if os.path.isdir(Olddir):
            continue
        Newdir = os.path.join(path, name+str(count+int(startNumber))+files[-4:])
        os.rename(Olddir, Newdir)
        count += 1
        print('正在重命名第{0}个文件'.format(count))


if __name__ == '__main__':
    textPath = 'C:/Users/oyyk/Desktop/URLs.txt'
    outputPath = 'C:/Users/oyyk/Desktop/image_output/'
    pre_url = r'http://168.160.158.231/files/'
    image_num = 0
    class_names = []

    # 创建文件夹
    if not os.path.exists(outputPath):
        os.mkdir(outputPath)

    #读取txt
    with open(textPath, 'r') as f:
        for class_name in f.readlines():
            class_names.append(class_name.rstrip('\n'))

    #查找图片
    for i in range(len(class_names)):
        temp = class_names[i]
        if re.search(pre_url, temp):
            img_src = temp
            response = requests.get(img_src)
            image = Image.open(BytesIO(response.content))
            file_type = '.jpg'
            if image.mode == 'RGBA':
                file_type = '.png'
            outfile_name = re.split('/|\?', img_src)
            outPath = outputPath + outfile_name[4] + '/'
            print('正在处理第{0}个图片链接'.format(image_num+1))
            image.save(outputPath + outfile_name[5] + file_type)
            image_num += 1
    rename(outputPath, 'Page', 1)