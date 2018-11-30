#!/usr/bin/env python3
import csv
import numpy as np
import matplotlib.pyplot as plt

HEROES_COUNT = 113
FEATURES_COUNT = 90 + HEROES_COUNT


def load_dataset(filename, test_part=10):
    dataset = []

    with open(filename) as iris_file:
        reader = csv.reader(iris_file)
        for row in reader:
            label = int(row[-5])
            features = list(map(lambda v: 0 if v == '' else int(float(v)), row[3:-6]))

            n_features = []
            team1 = []
            team2 = []
            heroes1 = [i for i in range(0, 33, 8)]
            heroes2 = [i for i in range(40, 73, 8)]

            for i, feature in enumerate(features):
                if i in heroes1:
                    team1.append(feature)
                    continue

                if i in heroes2:
                    team2.append(feature)
                    continue

                n_features.append(feature)

            teams = [0] * HEROES_COUNT

            for hero in team1:
                teams[hero] = 1

            for hero in team2:
                teams[hero] = -1

            dataset.append((np.array(n_features + teams), label))

    np.random.shuffle(dataset)
    test_size = len(dataset) // test_part

    return {'train': dataset[:-test_size], 'test': dataset[-test_size:]}


def get_probability(model, x):
    w, b = model

    y = np.dot(x, w) + b
    return 1 / (1 + np.exp(-y))


def predict(model, x):
    probability = get_probability(model, x)
    return 1 if probability > 0.5 else 0


def train(epochs_count, learning_rate, alpha, gamma, dataset, q_check=False):
    epochs = []
    accuracies = []
    objects_count = len(dataset)

    w = np.zeros(FEATURES_COUNT)
    b = 0
    sum_w = w
    sum_b = b
    uw = np.zeros(FEATURES_COUNT)
    ub = 0

    for t in range(epochs_count):
        x, y = dataset[t % objects_count]
        probability = get_probability((w, b), x)

        uw = gamma * uw + (probability - y) * x + 2 * alpha * w
        ub = gamma * ub + (probability - y) + 2 * alpha * b
        w = w - learning_rate * uw
        b = b - learning_rate * ub

        if q_check and t % 100 == 0:
            epochs.append(t)
            accuracies.append(test((w, b), dataset))

        sum_w += w
        sum_b += b

    if q_check:
        return (sum_w / epochs_count, sum_b / epochs_count), (epochs, accuracies)
    else:
        return sum_w / epochs_count, sum_b / epochs_count


def test(model, dataset):
    total = len(dataset)
    correct = 0

    for x, y in dataset:
        prediction = predict(model, x)

        if prediction == int(y):
            correct += 1

    return float(correct) / total


def k_fold(dataset, k=5):
    test_size = len(dataset) // k

    best_accuracy = 0
    best_params = (None, None, None, None)

    for epochs_count in range(1000, 1501, 100):
        for learning_rate in np.linspace(0.01, 0.05, num=3):
            for alpha in np.linspace(0, 1, num=6):
                for gamma in np.linspace(0.5, 0.99, num=3):
                    sum_accuracy = 0

                    for i in range(k):
                        head, test_dataset, tail = np.split(dataset, [i * test_size, (i + 1) * test_size])
                        train_dataset = np.concatenate((head, tail))

                        model = train(epochs_count, learning_rate, alpha, gamma, train_dataset)
                        sum_accuracy += test(model, test_dataset)

                    accuracy = sum_accuracy / k
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_params = (epochs_count, learning_rate, alpha, gamma)

    return best_params, best_accuracy


dataset = load_dataset('dota2.csv')

plt.title('Dota 2 match results')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

print('----------Dota 2 match results-----------')

(epochs_count, learning_rate, alpha, gamma), k_fold_accuracy = k_fold(dataset['train'])
print('K-fold best accuracy = {:.02f}% for epochs count = {}, learning rate = {}, alpha = {} and gamma = {}'
      .format(k_fold_accuracy * 100, epochs_count, learning_rate, alpha, gamma))

model, (epochs, accuracies) = train(epochs_count, learning_rate, alpha, gamma, dataset['train'], True)
plt.plot(epochs, accuracies)

accuracy = test(model, dataset['test'])
print('Accuracy = {:.02f}%'.format(accuracy * 100))

plt.show()
