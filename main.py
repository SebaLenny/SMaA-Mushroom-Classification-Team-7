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
            print(key + ': ' + str(value_frequency[key]) + ',\t', end='')
        print(']')
        print('\tValues percentages:\t', end='')
        print('[', end='')
        for key in unique_sorted:
            print(key + ': ' + "%.2f" % value_percentage[key] + '%,\t', end='')
        print(']', end='')
        print('')


if __name__ == "__main__":
    mushrooms_df = load_data_pandas()
    print("\n\n--Information about the dataset:\n")
    print(mushrooms_df.info())
    print("\n\n--Values of labels:\n")
    show_property_analysis(mushrooms_df)
