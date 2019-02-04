import pandas as pd

from _config import home_path
from big_ctg import load_ctg_cd_name_dic
from monophological_analysis import bag_of_noun
from naive_basyes import NaiveBayes


class ProfessionClass():
    def __init__(self, text):
        self.text = text
        # バイトル職種コード・名前対応辞書読み込み
        self.ctg_cd_name_dic = load_ctg_cd_name_dic()
        self.nb = NaiveBayes()

    # 専門職用の学習
    def profession_train(self):
        cd = ['8B', '8C']
        for code in cd:
            with open(home_path + '/resources/corpus/big_ctg/' + code + '.csv')as fr:
                documents = fr.readline()
                self.nb.train(documents, code)
        print('train finish')

    # 分類
    def classify(self):
        words = bag_of_noun(self.text.replace('仕事内容', ''))
        return self.ctg_cd_name_dic[self.nb.classifier(words)]


