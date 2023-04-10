import pandas as pd
from math import log2
from sklearn.metrics import accuracy_score, recall_score, precision_score
from tqdm import tqdm

print("training: reading training data")

# Read the csv file containing the training data
df = pd.read_csv('IMDB_train.csv')

# Convert all review strings into lowercase
df['review'] = df['review'].apply(str.lower)

# Tokenize the review string into individual words
df['review'] = df['review'].apply(str.split)


print("training: reading vocabulary data")

# Create a list of vocabulary from the "imdb.vocab" file
with open('imdb.vocab', 'r') as f:
    vocab = f.read().splitlines()

print("training: calculating prior probabilities")

# Create two dictionaries to store positive/negative class probability with word as key
pos_prob = {}
neg_prob = {}

pos_count = len(df[df['sentiment'] == 'pos'])
neg_count = len(df[df['sentiment'] == 'neg'])

# Calculate the prior probabilities of positive and negative classes
pos_prior = pos_count / len(df)
neg_prior = neg_count / len(df)


print("training: calculating likelihood probabilities")

# Calculate the likelihoods of each word given the positive and negative classes
for word in tqdm(vocab):
    # Count the number of reviews containing the word
    word_sentiments = df[(df['review'].str.contains(word, regex=False))]['sentiment']
    
    # Count the number of positive and negative reviews
    word_pos_count = word_sentiments.str.count("pos").sum()
    word_neg_count = len(word_sentiments) - word_pos_count # optimization instead of 'neg'
    
    # Calculate the probabilities using Laplace add-one smoothing
    pos_prob[word] = (word_pos_count + 1) / (pos_count + len(vocab))
    neg_prob[word] = (word_neg_count + 1) / (neg_count + len(vocab))

def predict_sentiment(review):
    """
    Predicts the sentiment of a review using the Naive Bayes model.
    
    Args:
        review: A string containing the text of the review.
    
    Returns:
        A string representing the predicted sentiment ('pos' or 'neg').
    """
    # Tokenize the review into individual words
    words = review.split()
    
    # Initialize the probabilities of positive and negative classes
    pos_p = 0 if pos_prior == 0 else log2(pos_prior)
    neg_p = 0 if neg_prior == 0 else log2(neg_prior)
    
    # Calculate the probabilities by adding the log likelihoods of each word
    for word in words:
        if word in vocab:
            pos_p += log2(pos_prob[word])
            neg_p += log2(neg_prob[word])
    
    # Return the prediction based on which probability is higher
    if pos_p > neg_p:
        return 'pos'
    else:
        return 'neg'

def evaluate(test_df):
    """
    Evaluates the performance of the Naive Bayes model on a test set.
    
    Args:
        test_df: A pandas DataFrame containing the test data.
    
    Returns:
        A tuple containing the accuracy, precision, and recall of the model on the test set.
    """
    
    tqdm.pandas()
    
    # Make predictions for each review in the test set
    y_pred = test_df['review'].progress_apply(predict_sentiment)
    
    # Calculate accuracy, precision, and recall
    accuracy = accuracy_score(test_df['sentiment'], y_pred)
    precision = precision_score(test_df['sentiment'], y_pred, pos_label='pos')
    recall = recall_score(test_df['sentiment'], y_pred, pos_label='pos')
    
    # Return the results
    return accuracy, precision, recall


print()
print("testing: reading testing data")

# Read the csv file containing the test data
test_df = pd.read_csv('IMDB_test.csv')


print("testing: evaluating model performance")

# Evalutate the training data
result = evaluate(test_df)

print(f"- Accuracy:  {result[0]:.3f}")
print(f"- Precision: {result[1]:.3f}")
print(f"- Recall:    {result[2]:.3f}")