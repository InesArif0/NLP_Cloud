import numpy as np
import textblob
from textblob import TextBlob

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

from wordcloud import WordCloud
import contractions
import unidecode
import nltk
import pickle

nltk.download('punkt')
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet

nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re


### Texte en minuscule
def lower_case_convertion(text):
    lower_text = text.lower()
    return lower_text


### Suppression des tags HTML
def remove_html_tags(text):
    html_pattern = r'<.*?>'
    without_html = re.sub(pattern=html_pattern, repl=' ', string=text)
    return without_html


# Suppression des URLS
def remove_urls(text):
    url_pattern = r'https?://\S+|www\.\S+'
    without_urls = re.sub(pattern=url_pattern, repl=' ', string=text)
    return without_urls


# ASCII
def accented_to_ascii(text):
    # apply unidecode function on text to convert
    # accented characters to ASCII values
    text = unidecode.unidecode(text)
    return text


### Contraction
def expand_contractions(text):
    ### We can use the dict : https://stackoverflow.com/questions/60901735/importerror-cannot-import-name-contraction-map-from-contractions
    return contractions.fix(text)


#######################################################################################################################################

### Suppresion de la ponctuation
import string


def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))


def preprocessing(text):
    text = lower_case_convertion(text)
    text = remove_html_tags(text)
    text = remove_urls(text)
    text = accented_to_ascii(text)
    text = expand_contractions(text)
    text = remove_punctuation(text)
    return text


def tag_words(sentence_tokenized):
    keywordSet = {"never", "nothing", "nowhere", "none", "not"}

    for word in sentence_tokenized:
        if (word in keywordSet) and (sentence_tokenized.index(word) < len(sentence_tokenized) - 1):
            sentence_tokenized[sentence_tokenized.index(word) + 1] = sentence_tokenized[
                                                                         sentence_tokenized.index(word) + 1] + '_NEG'
            sentence_tokenized.pop(sentence_tokenized.index(word))
    return sentence_tokenized


nltk.download('stopwords')
def remove_stop_word(text):
    stop_words = stopwords.words('english')
    tokens = word_tokenize(text)
    tokens_without_sw = [word for word in tokens if not word in stop_words]
    text = ' '.join(tokens)


def merge(dict1, dict2):
    for i in dict2.keys():
        dict1[i] = dict2[i]
    return dict1


def pos_tagging(text):
    ### POS_TAGGING
    lmtzr = WordNetLemmatizer()
    all_tag_words = {}
    tokens = word_tokenize(text)
    tagged = dict(nltk.pos_tag(tokens))
    all_tag_words = merge(tagged, all_tag_words)
    return all_tag_words


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''


# initialize lemmatizer object
lemma = WordNetLemmatizer()


### Lemmatize with POS tagging
def lemmatize(text, all_tag_words):
    tokens = ''.join(text).split()
    for i, token in enumerate(tokens):
        try:
            tokens[i] = lemma.lemmatize(token, get_wordnet_pos(all_tag_words[token]))
        except:
            pass
    text = ' '.join(tokens)
    return text


labels_dict = {
    0: "bad customer service at phone",
    1: "disappointed taste",
    2: "cold pizza",
    3: "not enough chicken",
    4: "bad food quality",
    5: "bad service",
    6: "bad burgers",
    7: "long wait",
    8: "bad experience on all levels",
    9: "bad experience at the bar",
    10: "problem with the delivery or the online order",
    11: "bad experience several times",
    12: "unorganized staff",
    13: "poor quality sushis",
    14: "dangerous place"
}

with open('model_entraîné_InesArif.pickle.circ', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer_InesArif.pickle.circ', 'rb') as f:
    vectorizer = pickle.load(f)


def predict(text: str, nb_features: int, labels_dict=labels_dict, blob=True):
    if blob == True:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        print(polarity)
        if polarity > 0:
            return polarity, None

        if polarity < 0:
            text = preprocessing(text)
            dict_pos = pos_tagging(text)
            text = lemmatize(text, dict_pos)
            text = [text]
            vect = vectorizer.transform(text)
            nmf_features = list(list(model.transform(vect))[0])
            
            nmf_features_copy = nmf_features
            indexes = []
            indexes.append(nmf_features.index(max(nmf_features_copy)))
            nmf_features_copy.pop(indexes[0])

            for i in range(nb_features - 1):
                index = nmf_features.index(max(nmf_features_copy))
                if index in indexes : 
                    array = np.array(nmf_features)
                    indices = np.where(array == 0.0)[0]
                    index = [ind for ind in indices and ind not in indexes][0]
                indexes.append(index)
                nmf_features_copy.pop(index)
            topics = []
            for i in indexes:
               topics.append(labels_dict[i])
            return polarity, topics


