import csv
import pandas as pd
if __name__ == '__main__':
    file_name_raw = 'D:/Users/oyyk/PycharmProjects/F_G_P/data/RawData/output/proto_Results.csv'
    data_frame = pd.DataFrame(data=None, columns=['class_name', 'file_num', 'index_num','is_support'])
    # file_name_raw = 'D:/Users/oyyk/PycharmProjects/F_G_P/data/RawData/output/proto_TestResults.csv'
    # data_frame = pd.DataFrame(data=None, columns=['loss', 'accuracy', 'precision', 'recall', 'f1-score'])
    # file_name_raw = 'D:/Users/oyyk/PycharmProjects/F_G_P/data/RawData/output/train_Results.csv'
    # data_frame = pd.DataFrame(data=None, columns=['train_acc', 'val_acc'])

    data_frame.to_csv(file_name_raw, index=None)
    # for i in range(14):
    #     #选择第一个文件作为合并后的文件
    #     file_name_raw = '../data/RawData/Muticlass_csv_1.csv'
    #     #循环处理之后的文件
    #     file_name_p = '../data/RawData/Muticlass_csv_' + str(i + 2) + '.csv'
    #     reader = csv.DictReader(open(file_name_p))
    #     header = reader.fieldnames
    #     with open(file_name_raw,'a',newline='') as csv_file:
    #         writer = csv.DictWriter(csv_file, fieldnames = header)
    #         writer.writerows(reader)