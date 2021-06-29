import re
from nltk.stem import WordNetLemmatizer
from .functional.wordnet import get_word_synonyms, get_stopword

class SentenceProcessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

        stop_words = get_stopword("english")
        stop_words.append("the")
        clean_stop_words = [self.clean_text(word,True) for word in stop_words]
        self.stop_words = set(stop_words+clean_stop_words)

    
    def clean_text(self, text, only_al = False):
        """
            Loại bỏ những char nhiễu và nhừng phần thừa trong một từ
        """

        if only_al:
            text = re.sub('[^a-zA-Z0-9 ]','',text)
        else:
            text = text.replace("."," . ")
            text = text.replace("-"," ")
            text = re.sub('[^a-zA-Z0-9. ]','',text)
        text = text.lower()
        text = [self.lemmatizer.lemmatize(token) for token in text.split()]
        text = [self.lemmatizer.lemmatize(token, "v") for token in text]
        text = " ".join(text)
        return text
    
    def get_bag_of_word(self, word):
        """
            Lấy những từ đồng nghĩa với từ đã cho (hiện tại chỉ cần lấy những từ đồng nghĩa vs một số)
        """

        if word.isdigit():
            return get_word_synonyms(word)
        else:
            return [word]

    def remove_stopword(self, text):
        """
            Loại bỏ những stopword khỏi một đoạn text
        """

        text = [w for w in text.split() if not w in self.stop_words] 
        return " ".join(text)
    
    def preprocess_answer(self, answers):
        """
            Thay giá trị những đáp án 'all of the above' bằng tổng hợp giá trị của tất cả những đáp án còn lại
        """
        tidx = -1
        fix_ans = ""
        for idx, ans in enumerate(answers):
            if ans.lower() == "all of the above":
                tidx = idx
            else:
                fix_ans += ans + " "
        if tidx != -1:
            answers[idx] = fix_ans
        return answers

    