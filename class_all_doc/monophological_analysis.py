import sys
# pip モジュールがインストールされるパスをPythonが見に行くようにします。
# (もしくは環境変数$PYTHONPATHにそのパスをあらかじめ記入しておきます。
sys.path.append('/usr/local/lib/python3.6/site-packages')

import MeCab
import re


def to_bag_of_words(text, duplication=True):
    #m = MeCab.Tagger("mecabrc")
    m = MeCab.Tagger("-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd")
    m.parse('')
    text = re.sub('【.+?】', "", text)
    text = re.sub('\d+', '0', text)
    text=re.sub(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-…]+', "", text)
    text = re.sub('\[.+?\]', "", text)
    text=re.sub('\n','',text)
    text=re.sub('\r','',text)
    node = m.parseToNode(text)
    keywords = []
    while node:
        morpheme = node.surface
        if duplication or morpheme not in keywords:
            keywords.append(morpheme)
        node = node.next
    return keywords


def bag_of_noun(text):
    #m = MeCab.Tagger("mecabrc")
    m = MeCab.Tagger("-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd")
    if text!="":

        m.parse('')
        text = re.sub('\d+', '0', text)
        text=re.sub(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-…]+', "", text)
        text = re.sub('\[.+?\]', "", text)
        text=re.sub('\n','',text)
        text=re.sub('\r','',text)
        #text = text.replace('・', '').replace('/', '').replace('系', '').replace('_','')
        node = m.parseToNode(text)
        # node = m.parseToNode(text.encode('utf-8'))
        keywords = []
        while node:
            if node.feature.split(",")[0] == "名詞":
                keywords.append(node.surface)
           
            node = node.next
        return keywords
