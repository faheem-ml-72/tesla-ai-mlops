
from textblob import TextBlob

def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

def get_sentiment_score(text):
    result = sentiment_model(text)[0]

    label = result['label']
    score = result['score']

    # Convert to numeric score
    if label == "positive":
        return score
    elif label == "negative":
        return -score
    else:
        return 0.0