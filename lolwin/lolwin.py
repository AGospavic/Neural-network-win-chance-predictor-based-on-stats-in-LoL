import numpy as np
import pandas as pd

datafile = "data.csv"
nonfirst_layers = [4,4,1]
epochs = 1000000
lr = 0.01

def read_data() -> pd.DataFrame:
    dataframe = pd.read_csv(datafile)
    return dataframe

def prepare_data(data : pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data_input : np.typing.NDArray[np.float64]
    data_output : np.typing.NDArray[np.float64]

    data_input = data[["Minutes","Kills","Deaths","Assists","CS"]].to_numpy()
    data_output = data[["Outcome"]].to_numpy()
    
    data_input = (data_input - data_input.mean(axis=0)) / data_input.std(axis=0)
    
    split = int(0.8 * data_input.shape[0])
    train_input, train_output = data_input[:split], data_output[:split]
    test_input, test_output = data_input[split:], data_output[split:]
    
    return train_input, train_output, test_input, test_output

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_backwards(x):
    return x * (1 - x)

def neural_network_config(X : np.ndarray) -> list[int]:
    return [X.shape[1],*nonfirst_layers]

def forward_calc(network_shape : list[int], X : np.ndarray, W : list[np.ndarray], b : np.ndarray) -> list[np.ndarray]:
    all_X = [X]
    for i in range(len(network_shape)-1):
        all_X = [*all_X, sigmoid(all_X[i] @ W[i] + b[i])]
    return all_X

def backward_calc(result : np.ndarray, X : list[np.ndarray], W : list[np.ndarray]) -> list[np.ndarray]:
    n = X[0].shape[0]
    dW = []
    db = []
    for i in range(len(X)-1,0,-1):
        result = result * sigmoid_backwards(X[i])
        dW = [(X[i-1].T @ result) / n, *dW]
        db = [np.sum(result, axis=0) / n, *db]
        if(i != 1):
            result = result @ W[i-1].T
    return dW, db

def neural_network_init(network_shape : list[int], X : np.ndarray, y : np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:

    rng = np.random.default_rng()
    W = [rng.standard_normal((network_shape[i],network_shape[i+1])) for i in range(len(network_shape)-1)]
    b = [np.zeros(network_shape[i+1]) for i in range(len(network_shape)-1)]
    n = X.shape[0]

    for epoch in range(epochs):
        all_X = forward_calc(network_shape, X, W, b)

        error = all_X[-1] - y
        #Loss function = sum(1/2 * error^2)

        dW, db = backward_calc(error, all_X, W)
        
        for i in range(len(network_shape)-1):
            W[i] -= lr * dW[i]
            b[i] -= lr * db[i]
    
    return W, b

def main() -> None:

    data = read_data()
    train_input, train_output, test_input, test_output = prepare_data(data)

    network_shape = neural_network_config(train_input)
    W, b = neural_network_init(network_shape, train_input, train_output)
    
    result = forward_calc(network_shape, test_input, W, b)[-1]
    for (_, row), predict in zip(data.iloc[int(0.8*data.shape[0]):].iterrows(), result):
        print(predict, row.values)
    
    error = 0
    for actual, predict in zip(test_output, result):
        error += abs(actual-predict)**2
    error = error / test_output.shape[0]

    return

main()