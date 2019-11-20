import numpy as np

data_path = 'Data/mushrooms.csv'


def load_data():
    data = []
    with open(data_path) as file:
        for line in file:
            data.append(line.strip().split(','))
    return np.array(data[0]), np.array(data[1:])


if __name__ == "__main__":
    y, X = load_data()
    print('shape of data: ', np.shape(X))
    print('labels: ', y)
    print('data: ', X)

