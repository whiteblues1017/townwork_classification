import pandas as pd

from _config import resources_path


def load_report_tw():
    df = pd.read_csv(resources_path + '/report_tw.csv', header=None, index_col=None, dtype=str, sep='\t')
    df = df.fillna(' ')
    return df


def extract_col():
    df = load_report_tw()
    job_mgr_no = df[0].tolist()
    job_content = df[3].tolist()
    title = df[4].tolist()
    big_ctg = df[19].tolist()
    documents = df.iloc[:, 73:76].values.tolist()
    print(documents)
    with open(resources_path + 'report_tw_ex.csv', 'w')as fw:
        fw.write('"job_mgr_no","job_content","title","big_ctg","mid_ctg","text"\n')
        for i in range(len(df)):
            # print(documents[i])
            fw.write('"' + job_mgr_no[i]
                     + '","' + job_content[i]
                     + '","' + title[i]
                     + '","' + big_ctg[i]
                     + '","' + ' '.join(documents[i]) + '"\n')


if __name__ == '__main__':
    extract_col()
