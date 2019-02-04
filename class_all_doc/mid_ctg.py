import multiprocessing
from multiprocessing.pool import Pool

import sys

from multiprocessing import Manager

sys.path.append('/usr/local/lib/python3.6/site-packages')

import pandas as pd
from _config import home_path, results_path
from big_rule_base import load_townwork_answer, BigCtg
from monophological_analysis import bag_of_noun
from naive_basyes import NaiveBayes


class MidCtg():
    def __init__(self, big_ctg):
        self.nb = NaiveBayes()
        self.big_ctg = big_ctg

        # バイトル職種コード・名前対応辞書読み込み
        self.ctg_cd_name_dic = load_ctg_cd_name_dic_mid()

    def train(self):
        cd = []
        for ctg_mid, ctg_big in self.ctg_cd_name_dic[1].items():
            if ctg_big == get_key_from_value(self.ctg_cd_name_dic[2], self.big_ctg):
                cd.append(str(ctg_mid))
        for code in cd:
            try:
                with open(home_path + '/resources/corpus/mid_ctg/' + code + '.csv')as fr:
                    documents = fr.readline()
                    self.nb.train(documents, code)

            except:
                c = 1

    # 分類
    def classify(self, text):
        words = bag_of_noun(text.replace('仕事内容', ''))
        if self.nb.classifier(words) is not None:
            return self.ctg_cd_name_dic[0][int(self.nb.classifier(words))]
        else:
            return 'None'


def split_list(l, n):
    """
    リストをサブリストに分割する
    :param l: リスト
    :param n: サブリストの要素数
    :return:
    """
    for idx in range(0, len(l), n):
        yield l[idx:idx + n]


def get_trained_nb_instance_dict(big_results):
    trained_nb_instance_dict = {}

    for ctg in list(list(set(big_results.values()))):
        print(ctg)
        mid_ctg = MidCtg(ctg)
        mid_ctg.train()
        trained_nb_instance_dict[ctg] = mid_ctg
    return trained_nb_instance_dict


class Multi():
    def __init__(self, nb_instances):
        self.nb_instances = nb_instances

    def split_bach_classification(self, split_dict):
        df = load_townwork_answer()
        mid_results = {}
        i = 0
        for media_code, big_ctg in split_dict.items():
            print(media_code)
            # progress(i, len(split_dict))
            text = df['ad_type'][df['media_code'] == media_code].to_string(index=None)
            # mid_ctg = MidCtg(big_ctg)
            # mid_ctg.train()
            mid_results[media_code] = self.nb_instances[big_ctg].classify(text)
            i += 1
        return mid_results


def multiprocess_classify(big_results):
    split_dicts = []
    nb_instances = get_trained_nb_instance_dict(big_results)
    print(big_results)
    for list in split_list([*big_results], int(len(big_results) / 4)):
        batch_dict = {}
        for key in list:
            batch_dict[key] = big_results[key]
        print(len(batch_dict))
        split_dicts.append(batch_dict)

    # 並列数を決めて、Poolを用意
    pool = Pool(4)
    # print('start')
    multi=Multi(nb_instances)
    # 並列処理実行
    print(split_dicts)
    multi_results = pool.map(multi.split_bach_classification, split_dicts)
    multi_results = {k: v for dic in multi_results for k, v in dic.items()}
    print(multi_results)

    """
    with Manager() as manager:
        # マネージャーから辞書型を生成します.
        multi_results = manager.dict()
        jobs = []
        for data in split_dicts:
            job = multiprocessing.Process(target=split_bach_classification, args=(data, nb_instances, multi_results))
            jobs.append(job)
            job.start()
        [job.join() for job in jobs]
        # multi_results = {k: v for dic in multi_results for k, v in dic.items()}
        export_mid_classify_result(multi_results, big_results, load_townwork_answer()) 
    """
    return multi_results


def progress(p, l):
    sys.stdout.write("\rclassify progress: %d  / 100" % (int(p * 100 / (l - 1))))
    sys.stdout.flush()


def get_key_from_value(d, val):
    keys = [k for k, v in d.items() if val == v]

    if keys:
        return keys[0]
    else:
        return None


def load_ctg_cd_name_dic_mid():
    df = pd.read_csv(home_path + '/resources/data/職種マスター.csv')
    shok_cd = df['shok_cd'].tolist()
    shok_name = df['shok_name'].tolist()
    ctg_cd = df['ctg_cd'].tolist()
    ctg_name = df['ctg_name'].tolist()

    dic = {}
    for i in range(len(df)):
        dic[shok_cd[i]] = shok_name[i]
    # 中カテゴリ-大カテゴリ対応
    dic_mid_big = {}
    for i in range(len(df)):
        dic_mid_big[shok_cd[i]] = ctg_cd[i]
    # 大カテゴリのcodeとname
    dic_big = {}
    for i in range(len(df)):
        dic_big[ctg_cd[i]] = ctg_name[i]

    return dic, dic_mid_big, dic_big


def export_mid_classify_result(mid_results, big_results, df):
    with open(results_path + '/townwork_rule_base_mid.csv', 'w')as fw:
        fw.write('"corporation_code","corporation_name","addressall","media_code","'
                 + 'job_category","job_category","media_name","big_classification_result","'
                 + 'mid_classification_result","ad_type"\n')

        for i in range(len(df)):
            fw.write('"' + df["corporation_code"][i] + '","' + df["corporation_name"][i] + '","' + df["addressall"][
                i] + '","'
                     + df["media_code"][i] + '","' + df["job_category"][i] + '","' + df["job_category"][i] + '","'
                     + df["media_name"][i] + '","' + big_results[df["media_code"][i]] + '","'
                     + mid_results[df["media_code"][i]] + '","' + df["ad_type"][i] + '"\n')


def main():
    result_df = pd.read_csv(results_path + '/townwork_rule_base_big.csv')
    big_results = dict(zip(result_df['media_code'].tolist(), result_df['job_category'].tolist()))
    df = load_townwork_answer()
    # mid_results ={}
    mid_results = multiprocess_classify(big_results)
    """
    print(load_ctg_cd_name_dic_mid())
    for media_code, big_ctg in big_results.items():
        text = df['ad_type'][df['media_code'] == media_code].to_string(index=None)

        if big_ctg=='軽作業・物流':
            print(media_code)
            mid_ctg = MidCtg(big_ctg)
            mid_ctg.train()
            mid_results[media_code] = mid_ctg.classify(text)
    """
    print(df)
    export_mid_classify_result(mid_results, big_results, df)


if __name__ == '__main__':
    main()
