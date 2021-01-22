import os
filePath = 'D:/Users/oyyk/PycharmProjects/F_G_P/data/RawData/'
dirlist = os.walk(filePath)
for i,j,k in os.walk(filePath):
    print(i,j,k)