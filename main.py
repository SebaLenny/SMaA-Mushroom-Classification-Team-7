import numpy as np
import pandas as pd

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
        unique_sorted = value_frequency[np.argsort(value_frequency)[::-1]].index.tolist()
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
            print(str(key) + ': ' + "%.2f" % value_percentage[key] + '%,\t', end='')
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


if __name__ == "__main__":
    mushrooms_df = load_data_pandas()
    # show_property_analysis(mushrooms_df) # data is for the most part defect free

    factorized_df = prepare_factorized_df(mushrooms_df)
    # show_property_analysis(factorized_df)

    one_hot_df = prepare_one_hot_df(mushrooms_df)
    # show_property_analysis(one_hot_df)
