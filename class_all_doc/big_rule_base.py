import pandas as pd

from _config import resources_path, results_path
from big_profession_ctg import ProfessionClass


class BigCtg():
    def __init__(self):
        # df = load_report_tw_ex()
        self.df = load_townwork_answer()
        self.baitoru_tw_big_ctg = baitoru_tw_big_ctg_dict()
        self.index_ctg_data_dict = dict(zip(self.df['media_code'].tolist(), self.df['job_category'].tolist()))

    def proffesion_classification(self,text):
        nb_big_ctg=ProfessionClass(text)
        nb_big_ctg.profession_train()
        result = nb_big_ctg.classify()
        return result

    def classification(self):
        class_result = {}
        for index, ctg in self.index_ctg_data_dict.items():
            ctg = ctg[ctg.find('_') + 1:]
            if ctg == '専門職/その他':
                class_result[index] = self.proffesion_classification(self.df['ad_type'][self.df['media_code']==index].to_string(index=None))
            else:
                class_result[index] = self.baitoru_tw_big_ctg[ctg]

        return class_result


def load_report_tw_ex():
    df = pd.read_csv(resources_path + '/report_tw_ex_test.csv', index_col=None, quotechar='"')
    df = df.fillna(' ')
    return df


def load_townwork_answer():
    df = pd.read_csv(resources_path + '/data/townwork_answer.csv', index_col=None)
    df = df.fillna(' ')
    return df


def baitoru_tw_big_ctg_dict():
    big_ctg_dict = {}
    df = pd.read_csv(resources_path + '/data/baitoru_tw.csv', quotechar='"', index_col=None)
    for i in range(len(df)):
        big_ctg_dict[df['townwork'][i]] = df['baitoru'][i]
    return big_ctg_dict


# 出力
def get_fw_str(results):
    fw_str = ''
    for index, classification_result in results.items():
        fw_str += '"' + str(index) + '","' + classification_result + '"\n'
    return fw_str


def export_results(results):
    with open(results_path + '/townwork_rule_base_big.csv', 'w')as fw:
        fw.write(get_fw_str(results))


if __name__ == '__main__':
    big_ctg = BigCtg()
    result = big_ctg.classification()
    export_results(result)
