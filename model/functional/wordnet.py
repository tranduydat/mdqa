from nltk.corpus import wordnet
from nltk.corpus import stopwords

def get_word_synonyms(word):
    """
        tìm những từ đồng nghĩa với từ đã cho
    """

    word_synonyms = set([word])
    for synset in wordnet.synsets(word):
        for lemma in synset.lemma_names():
            word_synonyms.add(lemma.lower())
    return list(word_synonyms)

def get_stopword(lang: str):
    """
        tìm những stopword của một ngôn ngữ chỉ định
    """

    return stopwords.words(lang)