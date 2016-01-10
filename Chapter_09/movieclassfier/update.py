import pickle
import sqlite3
import numpy as np 
import os
#import HashingVectorizer from local directory
from vectorizer import vect

def update_model(db_path, model, batch_size = 10000):
	conn = sqlite3.connect(db_path)
	c = conn.cursor()
	c.execute('SELECT * from review_db')
	results = c.fetchmany(batch_size)
	while results:
		data = np.array(results)
		X = data[:, 0]
		y = data[:, 1].astype(int)

		classes = np.array([0, 1])
		X_train = vect.transform(X)
		clf.partial_fit(X_train, y, classes = classes)
		results = c.fetchmany(batch_size)

	conn.close()
	return None

curr_dir = os.path.dirname(os.path.abspath(__file__))
pkl_file = os.path.join(curr_dir, 'pkl_objects', 'classifier.pkl')
with open(pkl_file, 'rb') as f:
	u = pickle._Unpickler(f)
	u.encoding = 'latin1'
	clf = u.load()
db = os.path.join(curr_dir, 'reviews.sqlite')

update_model(db_path = db, model = clf, batch_size = 10000)