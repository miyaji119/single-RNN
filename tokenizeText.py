# -*- coding: utf-8 -*-
# 将xlsx中每一行文本分词，并存为utf-8的csv
import csv,xlwt,xlrd
import jieba,jieba.analyse
import jieba.posseg as pesg


def token():
    # 导入自定义词典
    jieba.load_userdict('/home/ubuntu/文档/dicts/dict.txt')
    # 启用停止词词典
    stop_words_path = "/home/ubuntu/文档/dicts/extra_dict/stop_words.txt"
    jieba.analyse.set_stop_words(stop_words_path)
    stop_words = [line.strip() for line in open(stop_words_path, encoding='utf-8').readlines()]
    # 自定义语料库
    jieba.analyse.set_idf_path("/home/ubuntu/文档/dicts/extra_dict/idf.txt.big")

    fdir = '/home/ubuntu/文档/corpus/'
    # 源文件
    input_name = 'content.xlsx'
    # 保存文件
    output_name = 'tokenizedContent-utf8.csv'
    output_name1 = 'tokenizedContent&POS-utf8.csv'
    # 打开excel文件
    f = open(fdir+input_name, encoding='utf-8')
    # 读取csv文件
    # data = csv.reader(f)
    # 写入模式设置
    out = open(fdir+output_name, 'w', newline='', encoding='utf-8')
    csv_write = csv.writer(out, dialect='excel')
    out1 = open(fdir + output_name1, 'w', newline='', encoding='utf-8')
    csv_write1 = csv.writer(out1, dialect='excel')
    # 读取excel文件
    data = xlrd.open_workbook(fdir+input_name)
    # 获得第1张工作表
    table = data.sheets()[0]
    # 获得表的行数
    nrows = table.nrows

    # for content in data:
    for i in range(nrows):
        cell = table.cell(i, 0).value
        # print("row "+str(i)+": "+cell)

        # 分词+POS
        words = pesg.cut(cell)
        sent = ""
        pos = ""
        for word, flag in words:
            if word not in stop_words:
                sent += word + ';'
                pos += flag + ';'
        sent = sent[:-1]
        pos = pos[:-1]

        # 写入csv文件中
        csv_write.writerow([sent])  # 使用csv_write.writerow(cell)会将单词拆分成一个个字母存在一个单元格中
        csv_write1.writerow([pos])
    print("saved")
    # print('word_flag:', word_flag)

    # return word_flag

# token()

