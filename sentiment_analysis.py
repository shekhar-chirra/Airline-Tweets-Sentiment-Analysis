import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

import numpy as np
import pandas as pd
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import seaborn as sns
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from plotly.offline import plot
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import warnings

warnings.filterwarnings("ignore")

analyzer = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

ASPECTS = ['flight', 'service', 'delay', 'staff', 'luggage', 'customer service']

def preprocess_text(text):
    if isinstance(text, str):
        text = re.sub("[^a-zA-Z ]", '', text)
        text = text.lower().strip()
        return text
    else:
        return ''

def vader_sentiment_analysis(text):
    score = analyzer.polarity_scores(text)
    compound = score['compound']
    if compound >= 0.7:
        return 'Very Positive'
    elif compound >= 0.3:
        return 'Positive'
    elif compound > -0.3:
        return 'Neutral'
    elif compound > -0.7:
        return 'Negative'
    else:
        return 'Very Negative'

def aspect_sentiment_analysis(text, window=4):
    words = [w.rstrip('s') for w in text.split()]      # handles plurals: delays -> delay
    result = {a: [] for a in ASPECTS}
    for aspect in ASPECTS:
        a_words = [w.rstrip('s') for w in aspect.split()]
        n = len(a_words)
        for i in range(len(words) - n + 1):
            if words[i:i + n] == a_words:
                context = ' '.join(words[max(0, i - window): i + n + window])
                score = analyzer.polarity_scores(context)['compound']
                if score >= 0.05:
                    result[aspect].append('Positive')
                elif score <= -0.05:
                    result[aspect].append('Negative')
    return result
    
def detect_fake_review_single(review_text):
    review_length = len(review_text.split())
    positive_words = ['excellent', 'awesome', 'fantastic', 'amazing', 'great', 'love']
    overly_positive = any(word in review_text for word in positive_words)
    short_review = review_length < 5
    highly_generic = len(set(review_text.split())) < 3
    return overly_positive or short_review or highly_generic

def detect_fake_reviews_enhanced(df):
    df['review_length'] = df['reviewText'].apply(lambda x: len(x.split()))
    positive_words = ['excellent', 'awesome', 'fantastic', 'amazing', 'great', 'love']
    df['overly_positive'] = df['reviewText'].apply(lambda x: any(word in x for word in positive_words))
    df['short_review'] = df['review_length'] < 5
    df['highly_generic'] = df['reviewText'].apply(lambda x: len(set(x.split())) < 3)
    df['potential_fake'] = np.where(
        df['overly_positive'] | df['short_review'] | df['highly_generic'], True, False)
    return df

def plot_aspect_sentiments(df):
    aspect_counts = {a: {'Positive': 0, 'Negative': 0} for a in ASPECTS}
    for aspect_sentiments in df['aspect_sentiments']:
        for aspect, sentiments in aspect_sentiments.items():
            for sentiment in sentiments:
                aspect_counts[aspect][sentiment] += 1
    aspect_df = pd.DataFrame(aspect_counts).T          # aspects on the x-axis
    print("\nAspect sentiment counts:\n", aspect_df)
    aspect_df.plot(kind='bar', stacked=True, color=['#2ECC40', '#FF4136'])
    plt.title('Aspect-based Sentiment Analysis')
    plt.xlabel('Aspect')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
def plot_potential_fake_reviews(df):
    plt.figure(figsize=(4, 4))
    sns.countplot(x='potential_fake', data=df)
    plt.title('Potential Fake Reviews')
    plt.xlabel('Is Potential Fake')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

def categorical_variable_summary(df, column_name):
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('Countplot', 'Percentage'),
                        specs=[[{"type": "xy"}, {'type': 'domain'}]])
    fig.add_trace(go.Bar(
        y=df[column_name].value_counts().values.tolist(),
        x=[str(i) for i in df[column_name].value_counts().index],
        text=df[column_name].value_counts().values.tolist(),
        textfont=dict(size=14), name=column_name, textposition='auto', showlegend=False,
        marker=dict(color=['#FF4136', '#FF851B', '#FFDC00', '#2ECC40', '#0074D9'],
                    line=dict(color='#DBE6EC', width=1))), row=1, col=1)
    fig.add_trace(go.Pie(
        labels=df[column_name].value_counts().keys(),
        values=df[column_name].value_counts().values,
        textfont=dict(size=18), textposition='auto', showlegend=False, name=column_name,
        marker=dict(colors=['#FF4136', '#FF851B', '#FFDC00', '#2ECC40', '#0074D9'])), row=1, col=2)
    fig.update_layout(title={'text': column_name, 'y': 0.9, 'x': 0.5,
                             'xanchor': 'center', 'yanchor': 'top'},
                      template='plotly_white')
    plot(fig, filename=f'{column_name}_summary.html', auto_open=True)

def analyze_dataset(df):
    df['reviewText'] = df['reviewText'].map(preprocess_text)
    df['vader_sentiment'] = df['reviewText'].apply(vader_sentiment_analysis)
    df['aspect_sentiments'] = df['reviewText'].apply(aspect_sentiment_analysis)
    df = detect_fake_reviews_enhanced(df)
    print("\nVADER sentiment counts:\n", df['vader_sentiment'].value_counts())
    print("\nPotential fake counts:\n", df['potential_fake'].value_counts())
    categorical_variable_summary(df, 'vader_sentiment')
    plot_aspect_sentiments(df)
    plot_potential_fake_reviews(df)

def analyze_single_review(review_text):
    review_text_processed = preprocess_text(review_text)
    vader_sentiment = vader_sentiment_analysis(review_text_processed)
    aspect_sentiments = aspect_sentiment_analysis(review_text_processed)
    potential_fake = detect_fake_review_single(review_text_processed)
    print("\n*** Text Analysis Results ***")
    print(f"Text: {review_text}")
    print(f"VADER Sentiment: {vader_sentiment}")
    print(f"Potential Review: {'Yes' if potential_fake else 'No'}")

def input_review_data():
    while True:
        review_text = input("\nEnter a text review (or type 'exit' to quit): ")
        if review_text.lower() == 'exit':
            break
        analyze_single_review(review_text)

if __name__ == "__main__":
    # CSV path (raw string fixes the \U error). Change if you saved it elsewhere.
    df = pd.read_csv("Tweets.csv")
    df = df.rename(columns={"text": "reviewText"})
    df = df.dropna(subset=['reviewText'])
    analyze_dataset(df)
    input_review_data()