from athnlp.readers.brown_pos_corpus import BrownPosTag
import numpy as np

corpus = BrownPosTag()
dict_size = len(corpus.dictionary.x_dict.names)
label_size = len(corpus.dictionary.y_dict)
weights = [np.zeros(dict_size).reshape(-1, 1) for _ in range(0, label_size)]


def get_token_vector(token_index) -> np.array:
    token_vector = np.zeros((dict_size, 1))
    token_vector[token_index] = 1
    return token_vector


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

for epoch in range(1, 100):
    np.random.shuffle(corpus.train)
    for sentence in corpus.train:
        for token, label in zip(sentence.x, sentence.y):
            token_vector = get_token_vector(token)
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
            token_vector = get_token_vector(token)
            label_vector_gold = get_label_vector(label)
            label_gold = int(np.argmax(label_vector_gold))
            label_pred = int(np.argmax(predict_label_vector(token_vector)))
            if label_gold is not label_pred:
                error_count += 1
    accuracy_test = (token_count - error_count) / token_count

    token_count = 0
    error_count = 0
    for sentence in corpus.dev:
        for token, label in zip(sentence.x, sentence.y):
            token_count += 1
            token_vector = get_token_vector(token)
            label_vector_gold = get_label_vector(label)
            label_gold = int(np.argmax(label_vector_gold))
            label_pred = int(np.argmax(predict_label_vector(token_vector)))
            if label_gold is not label_pred:
                error_count += 1
    accuracy_dev = (token_count - error_count) / token_count

    print("epoch: {}, acc_test: {}, acc_dev: {}".format(epoch, accuracy_test, accuracy_dev))