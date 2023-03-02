from nltk import corpus, lm, tokenize
from nltk.util import ngrams

sents = []

for fileid in [
    "austen-emma.txt",
    "austen-persuasion.txt",
    "austen-sense.txt",
    "bible-kjv.txt",
    "blake-poems.txt",
    "bryant-stories.txt",
    "burgess-busterbrown.txt",
    "carroll-alice.txt",
    "chesterton-ball.txt",
    "chesterton-brown.txt",
    "chesterton-thursday.txt",
    "edgeworth-parents.txt",
    "melville-moby_dick.txt",
    "milton-paradise.txt",
    "shakespeare-caesar.txt",
    "shakespeare-hamlet.txt",
    "shakespeare-macbeth.txt",
    "whitman-leaves.txt",
]:
    # append load entire gutenberg corpus books tof tf-idf
    sents.extend(corpus.gutenberg.sents(fileid))


print(f"Using all books from nltk.corpus.gutenberg:")
print(f"- {len(sents)} sentences")
print()


def generate(model, length=12):
    generated = [
        # generate 3 sentences with randomized seeds
        model.generate(length, random_seed=x)
        for x in [2, 4, 42]
    ]

    for sent in generated:
        # convert tokenized sentence to string
        print(f"- {detokenize(sent)}")

    return generated


def detokenize(tokens):
    content = []
    for token in tokens:
        if token == "<s>" or token == "</s>":
            continue
        content.append(token)
    return tokenize.treebank.TreebankWordDetokenizer().detokenize(content)


def compute_perplexity(
    model,
    n,
    sents=[
        sent
        # sent.lower()  # turn all words into lower case
        for sent in [
            "He replied that he had not.",
            "You do not know what I suffer.",
            "Do you bite your thumb at us, sir?",
            "Forget to think of her.",
            "The white kitten had had nothing to do with it.",
            # sentences from original corpus of blake-poems
            "Where a thousand fighting men in ambush lie!",
            "Night and morning with my tears,",
            # sentences generated from entire gutenberg
            "my, They I ruminant of through, counsel, Have happened",  # unigram
            "made, and come to rock or a letter, and money",  # bigram
            "it, and come to public disgrace if Franklin had not had",  # trigram
        ]
    ],
):
    # convert sentences to padded words
    tokenized_sents = [
        list(
            ngrams(
                tokenize.word_tokenize(sent),
                n,
                pad_left=True,
                pad_right=True,
                left_pad_symbol="<s>",
                right_pad_symbol="</s>",
            )
        )
        for sent in sents
    ]

    # calculate perplexity of each sentence
    perplexity = [model.perplexity(sent) for sent in tokenized_sents]

    for index, sent in enumerate(sents):
        # sentence in ngrams with words in vocabulary
        lookup = model.vocab.lookup(tokenized_sents[index])

        # number of <UNK> tokens in ngrams
        unknowns = sum([ngram[-1] == "<UNK>" for ngram in lookup])
        # convert ngrams to string
        lookup_sent = detokenize([ngram[-1] for ngram in lookup])

        print(f"- {perplexity[index]:6.1f}, {unknowns} UNK, '{lookup_sent}'")

    return perplexity


def test_model(title, model, n):
    print(title)
    print("-" * len(title))

    print("Generate sentences (len=12)")
    generate(model)
    print()

    print("Compute perplexity")
    compute_perplexity(model, n)
    print()

    print()


""" Unigram
"""

# preprocess entire text with largest ngram length of 1
unigram_train, unigram_vocab = lm.preprocessing.padded_everygram_pipeline(1, sents)

lm_unigram = lm.MLE(1)  # using maximum likelihood estimator
lm_unigram.fit(unigram_train, unigram_vocab)

test_model("Unigram with MLE", lm_unigram, 1)

""" Bigram
"""

# preprocess entire text with largest ngram length of 2
bigram_train, bigram_vocab = lm.preprocessing.padded_everygram_pipeline(2, sents)

lm_bigram = lm.MLE(2)
lm_bigram.fit(bigram_train, bigram_vocab)

test_model("Bigram with MLE", lm_bigram, 2)

""" Trigram
"""

# preprocess entire text with largest ngram length of 3
trigram_train, trigram_vocab = lm.preprocessing.padded_everygram_pipeline(3, sents)

lm_trigram = lm.MLE(3)
lm_trigram.fit(trigram_train, trigram_vocab)

test_model("Trigram with MLE", lm_trigram, 3)
