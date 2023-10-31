# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 19:58:39 2023

@author: ASUS
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np


app = Flask(__name__)



model = joblib.load('C:/Users/ASUS/Downloads/best_model_amz.pkl')

# Defining a function to perform sentiment analysis
def predict_sentiment(text):
    if pd.isna(text):
        return 'null'
    else:
        predictions = model.predict([text]) 
        return predictions

@app.route('/')
def index():
    return render_template('indexx2.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' in request.files:
        # Handle file upload and analysis
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        if file:
            # Reading the CSV file 
            df = pd.read_csv(file)

        # Performing sentiment analysis
        sentiment_counts = df['reviewText'].apply(predict_sentiment).value_counts()

        # Create a pie chart
        labels = sentiment_counts.index
        sizes = sentiment_counts.values
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        plt.axis('equal')  
        pie_chart_path='static/pie_chart.png'
        plt.savefig(pie_chart_path)
        
        # Create a bar graph for overall ratings
        overall_counts = df['overall'].value_counts().sort_index()
        plt.figure(figsize=(8, 6))
        overall_counts.plot(kind='bar', color='cadetblue')
        plt.xlabel('Overall Ratings')
        plt.ylabel('Count')
        plt.title('Bar Graph of Overall Ratings')

        bar_graph_path = 'static/bar_graph.png'
        plt.savefig(bar_graph_path)
        

        

        total_reviews = len(df)

        return render_template('resultxx2.html', sentiment_counts=sentiment_counts, total_reviews=total_reviews, 
                                  pie_chart_path=pie_chart_path, bar_graph_path=bar_graph_path)


@app.route('/analyze-sentence', methods=['POST'])
def analyze_sentence():
    if 'sentence' in request.form:
        # Handle sentence analysis
        sentence = request.form['sentence']
        if not sentence.strip():
            return jsonify({'error': 'Please enter a sentence'})
        
        # Performing sentiment analysis on the entered sentence
        sentiment = predict_sentiment(sentence)
        
        
        sentiment_list = sentiment.tolist() if isinstance(sentiment, np.ndarray) else sentiment
        
        return render_template('sentiment_result.html', sentiment=sentiment_list)

if __name__ == '__main__':
    app.run(debug=False)
    

