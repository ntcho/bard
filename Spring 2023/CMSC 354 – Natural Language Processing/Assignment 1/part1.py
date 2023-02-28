from nltk import corpus, util, lm, tokenize
from nltk.util import ngrams

fileid = "blake-poems.txt"  # Using blake-poems
# fileid = "bible-kjv.txt"  # Using bible

words = corpus.gutenberg.words(fileid)
sents = corpus.gutenberg.sents(fileid)

print(f"Using {fileid} from nltk.corpus.gutenberg:")
print(f"- {len(words)} words, {len(sents)} sentences")
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
            # sentences generated from blake-poems
            "kiss' Near GIRL of me thee, because, FOUND dwelling", # unigram
            "heaven APPENDIX Prays Maker morn knits thee Deceit ancient Among Lyca clay", # unigram with tf-idf
            "human Brain . nibbled, sweeter smile, ON", # bigram
            "hand?", # trigram
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

""" Unigram with tf-idf
"""


class TfIdf(lm.api.LanguageModel):
    """Class for providing unigram model scores using tf-idf.

    Inherits initialization from BaseNgramModel.
    """

    def __init__(self, document_words, *args, **kwargs):
        """
        :param document_words: list of words of documents,
                               generated from LazyCorpusLoader.words().
        :type document_words: list of (list of str)
        """
        super().__init__(1, *args, **kwargs)

        # convert entire document into nltk.lm.Vocabulary
        self.document_words = document_words
        self.document_vocabs = [lm.Vocabulary(words) for words in document_words]

    def compute_tf(self):
        self.tf_dict = dict()
        vocab_len = sum(1 for _ in self.vocab)  # number of vocabulary in document

        for word in list(self.vocab):
            # tf = number of word f in document / number of words in document
            self.tf_dict[word] = self.vocab[word] / vocab_len
            # print(f"tf[{word}] = {self.tf_dict[word]}")

    def compute_idf(self):
        import math

        self.idf_dict = dict()
        N = len(self.document_vocabs)  # number of documents

        for word in list(self.vocab):
            # idf = log(number of documents / number of documents with word f)
            self.idf_dict[word] = math.log(
                N / sum([vocab[word] > 0 for vocab in self.document_vocabs])
            )
            # print(f"idf[{word}] = {self.idf_dict[word]}")

    def compute_tf_idf(self):
        self.compute_tf()
        self.compute_idf()

        self.tf_idf_dict = dict()

        for word in list(self.vocab):
            # tf-idf = tf * idf
            self.tf_idf_dict[word] = self.tf_dict[word] * self.idf_dict[word]

    def fit(self, *args, **kwargs):
        """Trains the model on a text.

        :param text: Training text as a sequence of sentences.
        """
        super().fit(*args, **kwargs)
        self.compute_tf_idf()

    def unmasked_score(self, word, context=None):
        """Returns the score for a word given a context using tf-idf.

        :param word: Word for which we want the score.
        """
        return self.tf_idf_dict[word]


# preprocess entire text with largest ngram length of 1
unigram_tf_idf_train, unigram_tf_idf_vocab = lm.preprocessing.padded_everygram_pipeline(
    1, sents
)

lm_unigram_tf_idf = TfIdf(
    [
        # load entire gutenberg corpus books tof tf-idf
        corpus.gutenberg.words("austen-emma.txt"),
        corpus.gutenberg.words("austen-persuasion.txt"),
        corpus.gutenberg.words("austen-sense.txt"),
        corpus.gutenberg.words("bible-kjv.txt"),
        corpus.gutenberg.words("blake-poems.txt"),
        corpus.gutenberg.words("bryant-stories.txt"),
        corpus.gutenberg.words("burgess-busterbrown.txt"),
        corpus.gutenberg.words("carroll-alice.txt"),
        corpus.gutenberg.words("chesterton-ball.txt"),
        corpus.gutenberg.words("chesterton-brown.txt"),
        corpus.gutenberg.words("chesterton-thursday.txt"),
        corpus.gutenberg.words("edgeworth-parents.txt"),
        corpus.gutenberg.words("melville-moby_dick.txt"),
        corpus.gutenberg.words("milton-paradise.txt"),
        corpus.gutenberg.words("shakespeare-caesar.txt"),
        corpus.gutenberg.words("shakespeare-hamlet.txt"),
        corpus.gutenberg.words("shakespeare-macbeth.txt"),
        corpus.gutenberg.words("whitman-leaves.txt"),
    ]
)
lm_unigram_tf_idf.fit(unigram_tf_idf_train, unigram_tf_idf_vocab)

test_model("Unigram with tf-idf", lm_unigram_tf_idf, 1)

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
