from tqdm import tqdm

from athnlp.readers.brown_pos_corpus import BrownPosTag
import numpy as np

corpus = BrownPosTag()
dict_size = len(corpus.dictionary.x_dict.names)
num_labels = len(corpus.dictionary.y_dict)
num_features = dict_size + num_labels
weights = [np.zeros(num_features).reshape(-1, 1) for _ in range(0, num_labels)]


def get_feature_vector_for_token(token_index, previous_token_label) -> np.array:
    # TODO IMPLEMENT
    token_vector = np.zeros((dict_size, 1))
    label_vector = np.zeros((num_labels, 1))
    token_vector[token_index] = 1
    if label_vector is not None:
        label_vector[previous_token_label] = 1
    return np.concatenate((token_vector, label_vector))


def get_label_vector(label_index) -> np.array:
    label_vector = np.zeros(num_labels)
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
    pred = np.zeros(num_labels)
    pred[np.argmax(label_scores)] = 1
    return pred


def update_weights(label_gold, label_pred, token_vector):
    weights[label_pred] -= token_vector
    weights[label_gold] += token_vector


def accuracy(correct, total):
    return correct / total

if __name__ == "__main__":
    for epoch in range(1, 11):
        np.random.shuffle(corpus.train)
        for sentence in tqdm(corpus.train[:1000], desc="training epoch {}".format(epoch)):




            previous_label = None
            for token, label in zip(sentence.x, sentence.y):
                for hyp_label in range(0, num_labels):
                    token_vector = get_feature_vector_for_token(token, previous_label)
                    label_gold = int(np.argmax(get_label_vector(label)))
                    label_pred = int(np.argmax(predict_label_vector(token_vector)))
                    if label_gold is not label_pred:
                        update_weights(label_gold, label_pred, token_vector)
                    previous_label = label_pred



            for hypothesis_idx in range(0, num_labels ** len(sentence.x)):

                token_idx = np.math.floor(hypothesis_idx / 12)

            # TODO ALL POSSIBLE HYPOTHESES
            for token_idx in range(0, len(sentence.x)):
                previous_label = None
                for label_idx in range(0, num_labels):
                    token_vector = get_feature_vector_for_token(token_idx, previous_label)
                    label_gold = int(np.argmax(get_label_vector(label)))
                    label_pred = int(np.argmax(predict_label_vector(token_vector)))
                    if label_gold is not label_pred:
                        update_weights(label_gold, label_pred, token_vector)
                    previous_label = label_pred



            for token, label in zip(sentence.x, sentence.y):
                token_vector = get_feature_vector_for_token(token, previous_label)

