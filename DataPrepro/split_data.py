import csv
import os


def split_csv(inpath, outpath, outname1, outname2, total_len, per):

    # 如果train.csv和vali.csv存在就删除
    # if os.path.exists('C:\\algo_file\\train.csv'):
    #     os.remove('C:\\algo_file\\train.csv')
    # if os.path.exists('C:\\algo_file\\vali.csv'):
    #     os.remove('C:\\algo_file\\vali.csv')

    with open(inpath, 'r', newline='') as file:
        csvreader = csv.reader(file)
        i = 0
        for row in csvreader:

            if i < round(total_len * per/100):
                # train.csv存放路径
                csv_path = os.path.join(outpath, outname1)
                print(csv_path)
                # 不存在此文件的时候，就创建
                if not os.path.exists(csv_path):
                    with open(csv_path, 'w', newline='') as file:
                        csvwriter = csv.writer(file)
                        csvwriter.writerow(row)
                    i += 1
                # 存在的时候就往里面添加
                else:
                    with open(csv_path, 'a', newline='') as file:
                        csvwriter = csv.writer(file)
                        csvwriter.writerow(row)
                    i += 1
            elif (i >= round(total_len * per/100)) and (i < total_len):
            	# vali.csv存放路径
                csv_path = os.path.join(outpath, outname2)
                print(csv_path)
                # 不存在此文件的时候，就创建
                if not os.path.exists(csv_path):
                    with open(csv_path, 'w', newline='') as file:
                        csvwriter = csv.writer(file)
                        csvwriter.writerow(row)
                    i += 1
                # 存在的时候就往里面添加
                else:
                    with open(csv_path, 'a', newline='') as file:
                        csvwriter = csv.writer(file)
                        csvwriter.writerow(row)
                    i += 1
            else:
                break

    print("训练集和验证集分离成功")
    return

if __name__ == '__main__':
    file_path = 'D:\\Users\\oyyk\\PycharmProjects\\F_G_P\\data\\Morris\\data__\\Muticlass_2_t\\'
    file_num=2
    j = 0
    k = 1
    for i in range(file_num):
        path = file_path+'Muticlass_csv_'+str(i)+'.csv'
        total_len = len(open(path, 'r').readlines())# csv文件行数
        per = 50 # 分割比例%
        outname1 = 'Muticlass_sv_'+str(j)+'.csv'
        outname2 = 'Muticlass_sv_'+str(k)+'.csv'
        split_csv(path,file_path,outname1,outname2, total_len, per)
        j = j+2
        k = k+2