from athnlp.readers.brown_pos_corpus import BrownPosTag
import numpy as np

corpus = BrownPosTag()
dict_size = len(corpus.dictionary.x_dict.names)
label_size = len(corpus.dictionary.y_dict)
weights = [np.random.rand(dict_size).reshape(-1, 1) for _ in range(0, label_size)]


def get_feature_vector_for_token(token_index, token_index_before=None, token_index_after=None) -> np.array:
    feature_vector = np.zeros((dict_size, 1))
    feature_vector[token_index] = 1
    if token_index_before != None:
        feature_vector[token_index_before] = 1
    if token_index_after != None:
        feature_vector[token_index_after] = 1
    return feature_vector


def get_label_vector(label_index) -> np.array:
    label_vector = np.zeros(label_size)
    label_vector[label_index] = 1
    return label_vector


def predict_label_vector(token_vector) -> np.array:
    """
    :param token_vector:
    :return: predicted label vector
    """
    # for label_weight_vector in weights[,:]:
    label_scores = []  # TODO DIRTY HACK, USE ARRAY FOR THIS
    for y, y_weights in enumerate(weights):
        score_y = np.dot(y_weights.transpose(), token_vector)
        label_scores.append(score_y)
    pred = np.zeros(label_size)
    pred[np.argmax(label_scores)] = 1
    return pred


def update_weights(label_gold, label_pred, token_vector):
    weights[label_pred] -= token_vector
    weights[label_gold] += token_vector

for epoch in range(1, 11):
    for sentence in corpus.train:
        for idx, (token, label) in enumerate(zip(sentence.x, sentence.y)):  # TODO SHUFFLE
            try:
                token_index_before = sentence.x[idx - 1]
            except (IndexError):
                token_index_before = None
            try:
                token_index_after = sentence.x[idx - 1]
            except (IndexError):
                token_index_after = None
            token_vector = get_feature_vector_for_token(token, token_index_before, token_index_after)
            label_vector_gold = get_label_vector(label)
            label_vector_pred = predict_label_vector(token_vector)
            label_gold = int(np.argmax(label_vector_gold))
            label_pred = int(np.argmax(label_vector_pred))
            if label_gold is not label_pred:
                update_weights(label_gold, label_pred, token_vector)
    ## EVAL
    token_count = 0
    error_count = 0
    for sentence in corpus.test:
        for token, label in zip(sentence.x, sentence.y):
            token_count += 1
            token_vector = get_feature_vector_for_token(token)
            label_vector_gold = get_label_vector(label)
            label_gold = int(np.argmax(label_vector_gold))
            label_pred = int(np.argmax(predict_label_vector(token_vector)))
            if label_gold is not label_pred:
                error_count += 1
    accuracy = (token_count - error_count) / token_count
    print("epoch: {}, accuracy: {}".format(epoch, accuracy))