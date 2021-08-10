

import numpy as np
import pandas as pd
import joblib
from waitress import serve
from flask import Flask, request, jsonify, render_template,redirect,url_for
import pickle
data=pd.read_csv(r'C:\Users\HP\Downloads\songs_complete_data.csv')
data1=pd.read_csv(r'C:\Users\HP\Downloads\final_dataset.csv')

app = Flask(__name__)
model = joblib.load(r'C:\Users\HP\Downloads\decision_tree.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/',methods=['GET','POST'])
def index():
    if request.method=='GET':
        key=sorted(data['Key'].unique())
        mode=sorted(data['Mode'].unique())
        genre=sorted(data1['Genre'].unique())
        explicit=sorted(data['explicit'].unique())
        return render_template('index.html',keys=key,modes=mode,genres=genre,explicit=explicit)
    
    
@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method=='POST':
        danceability=float(request.form['danceability'])
        energy=float(request.form['energy'])
        loudness=float(request.form['loudness'])
        speechiness=float(request.form['speechiness'])
        acousticness=float(request.form['acousticness'])
        liveness=float(request.form['liveness'])
        valence=float(request.form['valence'])
        tempo=float(request.form['tempo'])
        genre=int(request.form['genre'])
        input_variables=pd.dataframe([[danceability,energy,loudness,speechiness,acousticness,liveness,valence,tempo,genre]]) 
        prediction=model.predict(input_variables)[1]
        return render_template('index.html',prediction)
    
if __name__ == "__main__":
    
    app.run(debug=True)
    
    
    
    
    
        
        
    
    
    
        
        