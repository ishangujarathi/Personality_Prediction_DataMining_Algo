import pathlib

import numpy as np
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
import joblib
app = Flask(__name__)
model = joblib.load('train_model.pkl')
scaler = StandardScaler()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/submit', methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        age = request.form['age']
        openness = request.form['openness']
        neuroticism = request.form['neuroticism']
        conscientiousness = request.form['conscientiousness']
        agreeableness = request.form['agreeableness']
        extraversion = request.form['extraversion']
        female = request.form['isFemale']
        male = request.form['isMale']
        personality = str(model.predict([[int(age),int(openness),int(neuroticism),int(conscientiousness),int(agreeableness),int(extraversion),int(female),int(male)]]))
        return render_template("submit.html", answer=personality.replace('[', '').replace(']', '').replace("'", ""))


if __name__ == '__main__':
    app.run()
