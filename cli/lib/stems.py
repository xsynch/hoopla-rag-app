from nltk.stem import PorterStemmer



def get_stem_from_token(token):
    stemmer = PorterStemmer()
    return stemmer.stem(token)