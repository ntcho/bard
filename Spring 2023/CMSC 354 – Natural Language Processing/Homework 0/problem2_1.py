from nltk.tokenize import word_tokenize

text = ""

# read from file
with open("hw0.txt", "r") as file:
    text = file.read().replace("\n", "")

# tokenize entire text
tokens = word_tokenize(text)

print(tokens)
