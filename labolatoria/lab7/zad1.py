import nltk
import pandas as pd

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('all')

# text from article.txt
article_text = open('article.txt', 'r').read()

def preprocess_text(text):
    # Tokenize the text

    tokens = word_tokenize(text.lower())
    print("Liczba tokenów:", len(tokens))

    # Remove stop words

    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]
    print("Liczba tokenów po usunięciu stopwords:", len(filtered_tokens))

    # custom filter
    blacklist_token = [
        ""
    ]

    filtered_tokens = [token for token in filtered_tokens if token not in blacklist_token]
    print("Liczba tokenów po usunięciu custom:", len(filtered_tokens))


    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    print("Lematyzacja z pomocą WordNetLemmatizer")

    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    print("Liczba tokenów po lematyzacji:", len(lemmatized_tokens))

    processed_text = ' '.join(lemmatized_tokens)

    return processed_text

processed_article_text = preprocess_text(article_text)


from wordcloud import WordCloud
import matplotlib.pyplot as plt
wordcloud = WordCloud().generate(processed_article_text)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()