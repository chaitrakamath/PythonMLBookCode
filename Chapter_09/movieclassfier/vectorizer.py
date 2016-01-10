from sklearn.feature_extraction.text import HashingVectorizer
import re
import os
import pickle

cur_dir = os.path.dirname(__file__)
stop = pickle.load(open(os.path.join(cur_dir, 'pkl_objects', 'stopwords.pkl')
		, 'rb'))

def tokenizer(sent):
	sent = re.sub('<[^>]*>', '', sent)
	emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', sent.lower())
	sent = re.sub('[\W]', ' ', sent.lower()) + ''.join(emoticons).replace('-', '')
	tokenized = [w for w in sent.split() if w not in stop]
	return tokenized

vect = HashingVectorizer(decode_error = 'ignore', n_features = 2 ** 21, 
	preprocessor = None, tokenizer = tokenizer)