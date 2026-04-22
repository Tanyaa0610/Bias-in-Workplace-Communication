from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer

analyzer = SentimentIntensityAnalyzer()

def sentiment_score(text):
    return analyzer.polarity_scores(text)['compound']

def politeness_score(text):
    polite_words = ["please", "kindly", "thanks"]
    return sum([1 for w in text.split() if w in polite_words])

def get_tfidf(texts):
    vectorizer = TfidfVectorizer(max_features=3000)
    X = vectorizer.fit_transform(texts)
    return X, vectorizer

def normalize_score(score):
    return (score + 1) / 2