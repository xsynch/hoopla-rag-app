import string 


JSON_FILE = "data/movies.json"
STOPWORD_FILE = "data/stopwords.txt"



def removePunctuation(text:str):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def load_and_strip_stopwords(data):
    stop_word_list = ""
    stop_word_lines = ""
    with open(STOPWORD_FILE) as f:
        stop_word_list = f.read()
        stop_word_lines = stop_word_list.splitlines()
    if data in stop_word_lines:
        return None 
    else:
        return data 
    

def load_stop_words() -> list[str]:
        with open(STOPWORD_FILE) as f:        
            return f.read().splitlines()