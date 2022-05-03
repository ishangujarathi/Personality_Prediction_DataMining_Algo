import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

data = pd.read_csv('dataset.csv')

le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])
input_cols = ['Gender', 'Age', 'openness', 'neuroticism',
              'conscientiousness', 'agreeableness', 'extraversion']
output_cols = ['Personality (Class label)']
