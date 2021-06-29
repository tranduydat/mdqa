from sentence_transformers import SentenceTransformer, util
import torch
import pickle

class Sbert:
    """
        class để làm việc với sbert
    """
    def __init__(self, sbert: SentenceTransformer):
        """
            :param sbert: model sbert
        """

        self.sbert = sbert

    def vectorize(self, data):
        """
            chuyển text thuần sang vector bằng sbert
        """

        return self.sbert.encode(data)

    def vectorize_and_save(self, data: list, path: str):
        """
            chuyển text thuần sang vector bằng sbert và lưu
        """

        vector = self.vectorize(data)
        ret = {}

        ret["data"] = data
        ret["vector"] = vector

        with open(path,"wb") as f:
            pickle.dump(ret,f)

    def get_simsent_from_listsent(self, o_sent: str, l_sent: list, topn: int=10):
        """
            Cho vào một câu và tìm những câu gần giống câu đó nhất từ tập dữ liệu

            :param o_sent: câu đầu vào
            :param l_sent: tập những câu đã được encode bằng sbert
            :param topn: số lượng top câu trả về

            :return: index của top những câu giống câu đầu vào nhất
        """

        vector = self.vectorize([o_sent])

        score_box = []
        for v in l_sent:
            score = util.pytorch_cos_sim(vector, v)
            score_box.append(score.item())

        score_box = torch.tensor(score_box)
        sorted_index = score_box.argsort(descending=True)

        score_box = score_box[sorted_index]

        return sorted_index[0:topn], score_box
