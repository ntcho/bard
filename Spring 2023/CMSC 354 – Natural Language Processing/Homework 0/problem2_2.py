from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

text = ""
# text = "It was the best of times, it was the worst of times."

# read from file
with open("hw0.txt", "r") as file:
    text = file.read().replace("\n", "")

# tokenize entire text
tokens = word_tokenize(text)

# convert all tokens to lowercase
tokens = [token.lower() for token in tokens]

stopwords = stopwords.words("english")

tokens_filtered = []

for index, token in enumerate(tokens):
    # remove stopwords (e.g. it, the)
    if token not in stopwords:
        tokens_filtered.append(token)

print(tokens_filtered)
