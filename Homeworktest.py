import numpy as np
import random
#把10进制转换为2进制
def binary_encode(i,num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)][::-1])
#2进制转换为10进制
def binary_conv(i):
    return int(''.join(map(lambda x: str(int(x)), i)), 2)
#生成混沌序列
def seq_generator(length):
    x = random.uniform(-1,1)
    x_ = []
    for i in range(length):
        a = 1-2*x*x
        if a<0:
            x_.append(0)
        else:
            x_.append(1)
        x = a
    return np.array(x_)
#LSB替换
def LSB(seq,martix):
    s = np.zeros((martix.shape[0],martix.shape[1]),dtype=int)
    k = 0
    for i in range(martix.shape[0]):
        for j in range(martix.shape[1]):
            a = binary_encode(martix[i][j],8)
            if seq[k] != a[-1]:
                a[-1] = seq[i]
            s[i][j] = binary_conv(a)
            k = k+1
    return s

if __name__ == '__main__':
    #生成10*10随机矩阵
    A = np.random.randint(0,255,(10,10))
    a = seq_generator(100)
    A_LSB = LSB(a,A)
    print("\n原矩阵为：\n{}\n".format(A))
    print("需隐藏序列为：\n{}\n".format(a))
    print("嵌入后的矩阵为：\n{}\n".format(A_LSB))