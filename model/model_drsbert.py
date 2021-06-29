import torch
import pickle

from . import sentence_processor
from sentence_transformers import SentenceTransformer
from .doc_retrival import dr_sbert
from .functional.sentence_score import FixF1Score
from .functional.tokenize import QAtokenize

class QuestionAnswering:
    def __init__(self, sbert: SentenceTransformer, tokenizer, modeQA, topn: int=20):
        cuda_condition = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        self.modelQA = modeQA.to(self.device)

        self.sent_preprocessor = sentence_processor.SentenceProcessor()
        self.dr_sbert = dr_sbert.Sbert(sbert)

        self.tokenizer = tokenizer
        self.param = None
        self.topn = topn

    def load_file(self, path):
        with open(path,"r",encoding="utf-8") as f:
            raw_data = f.read()

        cleaned_data = self.sent_preprocessor.clean_text(raw_data).split(".")
        return cleaned_data

    def saveParam(self, source_path: str, destination_path: str):
        load_data = self.load_file(source_path)
        self.dr_sbert.vectorize_and_save(load_data, destination_path)

    def loadParam(self, path):
        with open(path,"rb") as f:
            self.param = pickle.load(f)

    def fixExistAnswer(self, ex_answers, mc_answers, mc_scores):    
        """
            Từ những đáp án khả thi tìm được và 4 đáp án đã cho, tìm ra đáp án đúng nhất
        """

        if len(mc_answers) == 0:
            return -1

        answer_max_score = 0
        answer_idx = 0
        for mc_answer, mc_score in zip(mc_answers,mc_scores):
            for ans_idx, ex_answer in enumerate(ex_answers):
                score = FixF1Score(ex_answer, mc_answer)
                score += mc_score
                if  score > answer_max_score:
                    answer_max_score = score
                    answer_idx = ans_idx
        return answer_idx
    
    def convert_ids_to_string(self, ids):
        return self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(ids))

    def QA(self, question: str):
        """
            tìm những đáp án khả thi từ trong dữ liệu đã cung cấp
        """

        sorted_index, score_box = self.dr_sbert.get_simsent_from_listsent(question, self.param["vector"], self.topn)

        all_text = []
        for idx in range(0,self.topn):
            target_idx = sorted_index[idx]
            text = ".".join(self.param["data"][target_idx:target_idx+3])
            all_text.append(text)

        input_ids, input_attention_mask = QAtokenize(self.tokenizer, question, all_text, 512)

        model_inputs = {"input_ids":input_ids.to(self.device),"attention_mask":input_attention_mask.to(self.device)}
        ids = input_ids.tolist()
        with torch.no_grad():
            all_answer_start_scores, all_answer_end_scores = self.modelQA(**model_inputs)
        
        all_answer_start = torch.argmax(all_answer_start_scores.cpu(),dim=-1)  
        all_answer_end = torch.argmax(all_answer_end_scores.cpu(),dim=-1)+1

        answers = []
        scores = []
        for idx in range(0,self.topn):
            answer = self.convert_ids_to_string(ids[idx][all_answer_start[idx]:all_answer_end[idx]])
            if answer != '' and "[CLS]" not in answer and "[SEP]" not in answer:
                answer = self.sent_preprocessor.remove_stopword(self.sent_preprocessor.clean_text(answer,True))
                answers.append(answer)
                scores.append(score_box[idx].item())

        return answers, scores

    def FindAnswer(self, question, answers):
        """
            tìm đáp án chính xác từ 4 đáp án đã cho
        """

        answers = self.sent_preprocessor.preprocess_answer(answers)
        answers = [self.sent_preprocessor.remove_stopword(self.sent_preprocessor.clean_text(ans,True)) for ans in answers]
        model_answer, scores = self.QA(question)
        return self.fixExistAnswer(answers,model_answer,scores), model_answer, scores