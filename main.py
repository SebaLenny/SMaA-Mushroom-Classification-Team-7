import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.metrics import f1_score
from sklearn import tree
from datetime import datetime
import graphviz

data_path = 'Data/mushrooms.csv'


def load_data_numpy():
    data = []
    with open(data_path) as file:
        for line in file:
            data.append(line.strip().split(','))
    return np.array(data[0]), np.array(data[1:])


def load_data_pandas():
    return pd.read_csv(data_path)


def show_property_analysis(df):
    for column in df.columns:
        unique_values = df[column].unique()
        unique_values_count = len(unique_values)
        value_frequency = df[column].value_counts()
        unique_sorted = value_frequency[np.argsort(
            value_frequency)[::-1]].index.tolist()
        value_percentage = value_frequency * 100 / len(df[column])

        print('')
        print(column)
        print("\tUnique values count:\t", end='')
        print(unique_values_count)
        print('\tPossible values:\t', end='')
        print(unique_values)
        print('\tValues frequencies:\t', end='')
        print('[', end='')
        for key in unique_sorted:
            print(str(key) + ': ' + str(value_frequency[key]) + ',\t', end='')
        print(']')
        print('\tValues percentages:\t', end='')
        print('[', end='')
        for key in unique_sorted:
            print(str(key) + ': ' +
                  "%.2f" % value_percentage[key] + '%,\t', end='')
        print(']', end='')
        print('')


def prepare_factorized_df(df):
    fact_df = df.copy()
    for column in df.columns:
        fact_df[column] = pd.factorize(mushrooms_df[column])[0]
    return fact_df


def prepare_one_hot_df(df):
    one_hot = pd.get_dummies(df)
    one_hot = one_hot.rename(columns={'class_e': 'class'})
    one_hot = one_hot.drop(columns=['class_p'])
    return one_hot


def train_test_split(df, split=.8):
    train = df.sample(frac=split)
    test = df.drop(train.index)
    return train, test


def knn_stats(training_data, test_data, k_n=51):
    train_labels, train_classes = training_data.values[:, 1:], training_data.values[:, 0]
    test_labels, test_classes = test_data.values[:, 1:], test_data.values[:, 0]
    ks, f1s = [], []
    time_start = datetime.now()
    for k in range(1, k_n):
        classifier = neighbors.KNeighborsClassifier(n_neighbors=k, p=1)
        classifier.fit(train_labels, train_classes)
        prediction = classifier.predict(test_labels)
        f1 = f1_score(test_classes, prediction)
        ks.append(k)
        f1s.append(f1)
    time_end = datetime.now()
    return ks, f1s, time_end - time_start


def knn_stats_p(training_data, test_data, p_steps=20, k=20):
    train_labels, train_classes = training_data.values[:, 1:], training_data.values[:, 0]
    test_labels, test_classes = test_data.values[:, 1:], test_data.values[:, 0]
    ps, f1s = [], []
    time_start = datetime.now()
    for p in np.linspace(1, 2, p_steps):
        classifier = neighbors.KNeighborsClassifier(n_neighbors=k, p=p)
        classifier.fit(train_labels, train_classes)
        prediction = classifier.predict(test_labels)
        f1 = f1_score(test_classes, prediction)
        ps.append(p)
        f1s.append(f1)
    time_end = datetime.now()
    return ps, f1s, time_end - time_start


def present_knn(k, a_score, b_score, a_label='', b_label='', x_label='', y_label=''):
    plt.plot(k, a_score, label=a_label)
    plt.plot(k, b_score, label=b_label)
    plt.legend()
    plt.xticks(k[::4])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    plt.show()


def tree_stats(training_data, test_data, train_size=-1, criterion='entropy', out_file=False):
    train_labels, train_classes = training_data.values[:train_size, 1:], training_data.values[:train_size, 0]
    test_labels, test_classes = test_data.values[:, 1:], test_data.values[:, 0]
    classifier = tree.DecisionTreeClassifier(criterion=criterion)
    classifier = classifier.fit(train_labels, train_classes)
    if out_file:
        tree_data = tree.export_graphviz(classifier,
                                         feature_names=training_data.columns[1:],
                                         class_names=['Poisonous', 'Edible'],
                                         rounded=True,
                                         )
        graph = graphviz.Source(tree_data)
        graph.render()
    prediction = classifier.predict(test_labels)
    return f1_score(test_classes, prediction)


def analyse_tree_training_size(train, test, criterion):
    f1s = []
    indx = []
    for i in range(5, 100):
        f1s.append(tree_stats(train, test, i, criterion))
        indx.append(i)
    for i in range(100, 500, 5):
        f1s.append(tree_stats(train, test, i, criterion))
        indx.append(i)
    for i in range(500, 2000, 15):
        f1s.append(tree_stats(train, test, i, criterion))
        indx.append(i)
    for i in range(2000, 6000, 100):
        f1s.append(tree_stats(train, test, i, criterion))
        indx.append(i)
    return indx, f1s


if __name__ == "__main__":
    mushrooms_df = load_data_pandas()
    mush_train, mush_test = train_test_split(mushrooms_df)
    # show_property_analysis(mushrooms_df) # data is for the most part defect free

    factorized_df = prepare_factorized_df(mushrooms_df)
    fact_train, fact_test = train_test_split(factorized_df)
    # show_property_analysis(factorized_df)

    one_hot_df = prepare_one_hot_df(mushrooms_df)
    oh_train, oh_test = train_test_split(one_hot_df)
    # show_property_analysis(one_hot_df)

    # KNN
    knn_k, knn_f1_normal, fact_delta = knn_stats(fact_train, fact_test)
    print(fact_delta)
    _, knn_f1_one_hot, oh_delta = knn_stats(oh_train, oh_test)
    print(oh_delta)
    present_knn(knn_k, knn_f1_normal, knn_f1_one_hot, a_label='Factorized data', b_label='One hot data', x_label='K',
                y_label='F1 score')

    # KNN - P
    p_ss, p_f1, p_deltas = knn_stats_p(fact_train, fact_test)
    plt.plot(p_ss, p_f1)
    plt.xlabel('P value')
    plt.ylabel('F1 score')
    plt.show()

    # DC
    indx, f1s = analyse_tree_training_size(oh_train, oh_test, 'entropy')
    plt.plot(indx, f1s)
    indx, f1s = analyse_tree_training_size(oh_train, oh_test, 'gini')
    plt.plot(indx, f1s)
    plt.show()
    tree_stats(oh_train, oh_test, out_file=True)
