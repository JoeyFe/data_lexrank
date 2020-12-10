import re
from lexrank import LexRank
from lexrank.mappings.stopwords import STOPWORDS
from path import Path
from typing import Any, Dict, List, Tuple
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import RegexpTokenizer


def model_sdk(test_path):
    TOKENIZER = RegexpTokenizer(r"\w+")
    LEMMATIZER = WordNetLemmatizer()
    STEMMER = PorterStemmer()

    def _tokenize_text(text: str) -> List[str]:
        def filter_by_pos(token: List[str]) -> List[str]:  # filter by pos tags
            return [t for t, pos in pos_tag(token) if
                    pos in ["NN", "NNP", "NNS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]]

        def remove_stopwords(token: List[str]) -> List[str]:
            return [w for w in token if not w in STOPWORDS]

        def tokenize_text(text: str) -> List[str]:
            return TOKENIZER.tokenize(text)  # list of token without punctuation etc.

        def get_token_lemmata(token: List[str]) -> List[str]:  # prefer over stemming
            return [LEMMATIZER.lemmatize(t) for t in token]

        def get_token_stems(token: List[str]) -> List[str]:
            return [STEMMER.stem(t) for t in token]

        token: List[str] = get_token_lemmata(remove_stopwords(tokenize_text(text.lower())))
        token = filter_by_pos(token)

        return token

    documents = []

    documents_dir = Path('../dataset/source_texts')

    for file_path in documents_dir.files('*.txt'):
        with file_path.open(mode='rt', encoding='utf-8') as fp:
            documents.append(fp.readlines())

    lxr = LexRank(documents, stopwords=STOPWORDS['en'])

    # with open("/dataset/source_texts/test.txt", encoding='utf-8') as f:
    #     loaded_text: str = f.read()
    with open(test_path, encoding='utf-8') as f:
        loaded_text: str = f.read()
    loaded_text = loaded_text.replace("\n\n", " ")

    # ****************************************************
    #
    # text pre-processing
    #
    # ****************************************************
    # wikipedia specific: remove paranthesis and brackets incl. text inside i.e. [1] or (28 June 1491 – 28 January 1547)
    for x in set(re.findall("[\(\[].*?[\)\]]", loaded_text)):
        loaded_text = loaded_text.replace(x, "")
    loaded_text = " ".join(loaded_text.split()).strip(".")  # remove double space

    # ***
    # split into sentences
    # use for clean-up and later to match best results with original sentences
    sentences: List[str] = loaded_text.split(".")
    for i, s in enumerate(sentences):
        sentences[i] = f"{s}.".strip()
    # ***
    # get cleaned version of each sentence
    cleaned_sentences: List[str] = [" ".join(_tokenize_text(s)) for s in sentences]  # list of sentences
    # get the n % most relevant sentences
    percentage_text_reduction = 30
    num_sentences = int((percentage_text_reduction * len(sentences)) / 100)
    # get summary with classical LexRank algorithm
    summary = lxr.get_summary(sentences, summary_size=num_sentences, threshold=.1)
    print(summary)

    # get summary with continuous LexRank
    # summary_cont = lxr.get_summary(sentences, threshold=None)
    # print(summary_cont)

    # get LexRank scores for sentences
    # 'fast_power_method' speeds up the calculation, but requires more RAM
    scores_cont = lxr.rank_sentences(
        sentences,
        threshold=None,
        fast_power_method=False,
    )
    print(scores_cont)
    with open("../dataset/save/summary.txt", 'w', encoding='utf-8') as f:
        for sentence in summary:
            f.write(sentence)
        f.write('\r\n')
        f.write('LexRank scores for sentences:')
        f.write('\r\n')
        f.write(str(scores_cont))
        f.close()
