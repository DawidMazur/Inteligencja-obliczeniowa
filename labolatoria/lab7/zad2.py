

opinia_pozytywna="We liked the location,easy access to the building and the manager problem solving skills behaviour and attitude. Would be a great apartmant in overall"
opinia_negatywna=" lack of cleanliness of bedding set/bed/mattress->bites to the morning , staff belie attitude through the phone and online chat. We decided to leave the apartment and booked in another. Still recovering after a week. better if i dont upload photos."

opinie = [
    opinia_pozytywna,
    opinia_negatywna
]

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import text2emotion as te

def preprocess_text(text):
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

def test_opinions(loop):
    sid = SentimentIntensityAnalyzer()
    for opinia in loop:
        print(opinia)
        print(sid.polarity_scores(preprocess_text(opinia)))
        print(te.get_emotion(opinia))
        print("")

test_opinions(opinie)

stronger_opinions = [
    "The apartment was very clean and comfortable. The location was great, close to the city center and the metro station. The host was very helpful and friendly. We had a great time in Warsaw and we would definitely recommend this apartment. Thank you!",
    "This was the absolute worst experience I have ever had with a hotel. The room was dirty, the staff was rude, and the location was terrible. I would not recommend this hotel to anyone. I would give it zero stars if I could. I will never stay here again. I would rather sleep in my car than stay here again.",
]

test_opinions(stronger_opinions)