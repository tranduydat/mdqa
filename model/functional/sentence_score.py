import math
from .search_word_in_sent import perfectSearch 

def FixF1Score(sent1: str, sent2: str):
    """
        Hàm tính điểm số tương đồng giữa hai câu

        :param sent1: câu số 1
        :param sent2: câu số 2
        :return: điểm số tương đồng
    """
    
    if len(sent1) == 0 or len(sent2) == 0:
        return 0
    sent1 = sent1.split()

    sent2_len = len(sent2.split())
    similiar = 0
    for word in sent1:
        tof, fix_sent2_len = perfectSearch(word, sent2)
        sent2_len -= fix_sent2_len
        if tof:
            similiar += 1
                        
    recall = similiar/len(sent1)
    precision = math.pow(similiar/sent2_len,1/6)
    
    if recall == 0 or precision == 0:
        return 0

    return (2*recall*precision)/(recall+precision)