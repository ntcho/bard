import pandas as pd
from math import log2
from sklearn.metrics import accuracy_score, recall_score, precision_score
from tqdm import tqdm

print("training: reading training data")

# Read the csv file containing the training data
df = pd.read_csv("IMDB_train.csv")

# Convert all review strings into lowercase
df["review"] = df["review"].apply(str.lower)

# Tokenize the review string into individual words
df["review"] = df["review"].apply(str.split)


print("training: reading vocabulary data")

# Create a list of vocabulary from the "imdb.vocab" file
with open("imdb.vocab", "r") as f:
    vocab = f.read().splitlines()

print("training: calculating prior probabilities")

# Create two dictionaries to store positive/negative class probability with word as key
pos_prob = {}
neg_prob = {}

# Create two dictionaries to store positive/negative occurence with word as key
pos_count = {word: 0 for word in vocab}
neg_count = {word: 0 for word in vocab}

pos_count_all = len(df[df["sentiment"] == "pos"])
neg_count_all = len(df[df["sentiment"] == "neg"])

# Calculate the prior probabilities of positive and negative classes
pos_prior = pos_count_all / len(df)
neg_prior = neg_count_all / len(df)


print("training: calculating likelihood probabilities (iterating through all reviews)")

# Helper function to optimize performance
def count_pos(word):
    pos_count[word] += 1

def count_neg(word):
    neg_count[word] += 1


for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    # Defines which operation to perform according to the label
    count = count_pos if row["sentiment"] == "pos" else count_neg

    for word in row["review"]:
        try:
            # Count the number of reviews containing the word
            count(word)
        except:
            # Ignore words that are not in the vocabulary
            continue
        
        
print("training: calculating likelihood probabilities (iterating through all vocabulary words)")

for word in tqdm(vocab):
    # Calculate the probabilities using Laplace add-one smoothing
    pos_prob[word] = (pos_count[word] + 1) / (pos_count_all + len(vocab))
    neg_prob[word] = (neg_count[word] + 1) / (neg_count_all + len(vocab))


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
    pos_p = log2(pos_prior)
    neg_p = log2(neg_prior)

    # Calculate the probabilities by adding the log likelihoods of each word
    for word in words:
        try:
            pos_p += log2(pos_prob[word])
            neg_p += log2(neg_prob[word])
        except:
            # Ignore words that are not in the vocabulary
            continue

    # Return the prediction based on which probability is higher
    if pos_p > neg_p:
        return "pos"
    else:
        return "neg"


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
    y_pred = test_df["review"].progress_apply(predict_sentiment)

    # Calculate accuracy, precision, and recall
    accuracy = accuracy_score(test_df["sentiment"], y_pred)
    precision = precision_score(test_df["sentiment"], y_pred, pos_label="pos")
    recall = recall_score(test_df["sentiment"], y_pred, pos_label="pos")

    # Return the results
    return accuracy, precision, recall


print()
print("testing: reading testing data")

# Read the csv file containing the test data
test_df = pd.read_csv("IMDB_test.csv")


print("testing: evaluating model performance")

# Evalutate the training data
result = evaluate(test_df)

print(f"- Accuracy:  {result[0]:.3f}")
print(f"- Precision: {result[1]:.3f}")
print(f"- Recall:    {result[2]:.3f}")
