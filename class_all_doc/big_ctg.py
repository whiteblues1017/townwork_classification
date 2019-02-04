from multiprocessing.pool import Pool

import sys

import gc

sys.path.append('/usr/local/lib/python3.6/site-packages')

import pandas
import pandas as pd

from _config import home_path, resources_path, results_path
from monophological_analysis import bag_of_noun
from naive_basyes import NaiveBayes


class Big_ctg():
    def __init__(self, no_text_dict):

        self.nb = NaiveBayes()
        self.text = list(no_text_dict.values())
        self.classify_len = len(no_text_dict)
        # バイトル職種コード・名前対応辞書読み込み
        self.ctg_cd_name_dic = load_ctg_cd_name_dic()

    # 学習
    def train(self):
        cd = ['8A', '8B', '8C', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89']
        for code in cd:
            with open(home_path + '/resources/corpus/big_ctg/' + code + '.csv')as fr:
                documents = fr.readline()
                self.nb.train(documents, code)
        print('train finish')

    # 分類
    def classify(self, index):
        class_dic = {}
        for i in index:
            # if 0 in index:
            #   progress(i, len(index))
            words = bag_of_noun(self.text[i].replace('仕事内容', ''))
            class_dic[i] = self.ctg_cd_name_dic[self.nb.classifier(words)]
        fw_str=get_fw_str(class_dic)
        del class_dic
        gc.collect()
        return fw_str

    def multiprocess_classify(self):
        all_index = list(range(self.classify_len))
        split_list = [all_index[i:i + int(len(all_index) / 4)] for i in
                      range(0, len(all_index), int(len(all_index) / 4))]

        # 並列数を決めて、Poolを用意
        pool = Pool(8)

        # 並列処理実行
        multi_results = pool.map(self.classify, split_list)

        return multi_results


def load_ctg_cd_name_dic():
    df = pandas.read_csv(home_path + '/resources/data/職種マスター.csv')
    ctg_cd = df['ctg_cd'].tolist()
    ctg_name = df['ctg_name'].tolist()

    dic = {}
    for i in range(len(df)):
        dic[ctg_cd[i]] = ctg_name[i]

    return dic


def progress(p, l):
    sys.stdout.write("\rclassify progress: %d  / 100" % (int(p * 100 / (l - 1))))
    sys.stdout.flush()


def load_report_tw_ex():
    df = pd.read_csv(resources_path + '/report_tw_ex_test.csv', index_col=None, quotechar='"')
    df = df.fillna(' ')
    return df


def generate_no_text_dict():
    df = load_report_tw_ex()
    dict = {}
    for i in range(len(df)):
        dict[df['job_mgr_no'][i]] = df['title'][i]+df['text'][i]
    return dict


# 出力
def get_fw_str(results):
    fw_str = ''
    for index, classification_result in results.items():
        fw_str += '"' + str(index) + '","' + classification_result + '"\n'
    return fw_str


def export_results(results):
    with open(results_path + '/report_tw_ex_result.csv', 'w')as fw:
        for result  in results:
            fw.write(result)


if __name__ == '__main__':
    big_ctg = Big_ctg(generate_no_text_dict())
    big_ctg.train()
    results = big_ctg.multiprocess_classify()
    export_results(results)
