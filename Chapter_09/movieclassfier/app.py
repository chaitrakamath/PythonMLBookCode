from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import pickle
import sqlite3
import os
import numpy as np
#import HashingVectorizer from local directory
from vectorizer import vect 
#import in order to update the model in event of crash
from update import update_model

app = Flask(__name__)

#PART1: Prepare classifier
#load classifier from pickle
curr_dir = os.path.dirname(os.path.abspath(__file__))
pkl_file = os.path.join(curr_dir, 'pkl_objects', 'classifier.pkl')
with open(pkl_file, 'rb') as f:
	u = pickle._Unpickler(f)
	u.encoding = 'latin1'
	clf = u.load()
db = os.path.join(curr_dir, 'reviews.sqlite')


def classify(document):
	label = {0: 'negative', 1: 'positive'}
	X = vect.transform([document])
	y = clf.predict(X)[0]
	prob = np.max(clf.predict_proba(X))
	return label[y], prob


def train(document, y):
	X = vect.transform([document])
	clf.partial_fit(X, [y])

def sqlite_entry(path, document, y):
	conn = sqlite3.connect(path)
	c = conn.cursor()
	c.execute("INSERT INTO review_db (review, sentiment, date)"\
		"VALUES(?, ?, DATETIME('now'))", (document, y))
	conn.commit()
	conn.close()


#PART2: Flask 
class ReviewForm(Form):
	moviereview = TextAreaField('', [validators.DataRequired(), 
										validators.length(min = 15)])

@app.route('/')
def index():
	form = ReviewForm(request.form)
	return render_template('reviewform.html', form = form)


@app.route('/results', methods = ['POST'])
def results():
	form = ReviewForm(request.form)
	if request.method == 'POST' and form.validate():
		review = request.form['moviereview']
		y, prob = classify(review)
		return render_template('results.html', content = review, 
			prediction = y, probability = round(prob*100, 2))
	return render_template('reviewform.html', form = form)

@app.route('/thanks', methods = ['POST'])
def feedback():
	feedback = request.form['feedback_button']
	review = request.form['review']
	prediction = request.form['prediction']

	inv_label = {'negative': 0, 'positive': 1}
	y = inv_label[prediction]
	if feedback == 'Incorrect':
		y = int(not(y))
	train(review, y)
	sqlite_entry(db, review, y)
	return render_template('thanks.html')

if __name__ == '__main__':
	update_model(filepath = db, model = clf, batch_size = 10000)