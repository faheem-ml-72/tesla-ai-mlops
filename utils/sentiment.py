
from textblob import TextBlob

def get_sentiment_score(text):
    polarity = TextBlob(text).sentiment.polarity
    
    # Convert -1 → 1  into 0 → 1
    normalized_score = (polarity + 1) / 2
    
    return normalized_score