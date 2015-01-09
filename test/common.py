
# some test data
corpus = ["he went down to the store".split(),
        "he needed a shovel from the store to shovel the snow".split()]
corpus_update = ["the snow was five feet deep".split()]

corpus_unigrams = {
 'a': [1, 1],
 'deep': [1, 1],
 'down': [1, 1],
 'feet': [1, 1],
 'five': [1, 1],
 'from': [1, 1],
 'he': [2, 2],
 'needed': [1, 1],
 'shovel': [2, 1],
 'snow': [2, 2],
 'store': [2, 2],
 'the': [4, 3],
 'to': [2, 2],
 'was': [1, 1],
 'went': [1, 1],
}
corpus_ndocs = 3

query = ["buy", "snow", "shovel", "shovel"]
document = ["the", "store", "sells", "snow", "shovel", "snow"]



