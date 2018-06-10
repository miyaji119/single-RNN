# -*- coding: utf-8 -*-
# 保存loss和学习率
import csv, xlwt, xlrd

def log_save(data):
    fdir = '/home/ubuntu/文档/'
    output = 'loss-log-0525-1420.xlsx'

    # 写入模式设置
    out = open(fdir+output, 'w', newline='', encoding='utf-8')
    csv_write = csv.writer(out, dialect='excel')

    # 写入csv文件中
    for cell in data:
        csv_write.writerow([cell[0],cell[1]])
    print("log saved "+output)

# log_save([[0.1, 0.01], [0.09, 3.02]])