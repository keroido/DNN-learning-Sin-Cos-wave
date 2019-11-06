import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

df = pd.read_csv('../input/data.csv')
df = df.drop('Unnamed: 0', axis=1)

X = df.iloc[:, :2]
y = df.iloc[:, 2:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

hidden_layer_sizes = [(10, 10,), (50, 50,), (100, 100,), (150, 150,), (200, 200,),
                      (10, 10, 10,), (50, 50, 50,), (100, 100, 100,), (150, 150, 150,), (200, 200, 200,),
                      (10, 10, 10, 10,), (50, 50, 50, 50,), (100, 100, 100, 100,), (150, 150, 150, 150,), (200, 200, 200, 200,)]
ln = list(range(len(hidden_layer_sizes)))
score_df = pd.DataFrame(columns={'sin_mse', 'cos_mse', 'r^2_score'})

for i, hidden_layer_size in zip(ln, hidden_layer_sizes):
    model = MLPRegressor(activation='relu', alpha=0, batch_size=100,
                         hidden_layer_sizes=hidden_layer_size, learning_rate_init=0.03,
                         random_state=0, verbose=False, solver='adam')
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    pred = pd.DataFrame(pred)
    x = pd.DataFrame({'x':(X_test['x0'] + X_test['x1']).tolist()})
    tes = y_test.rename(columns={'sin': 'sin(label)', 'cos': 'cos(label)'}).reset_index(drop=True)
    pre = pred.rename(columns={0: 'sin(prediction)', 1: 'cos(prediction)'}).reset_index(drop=True)
    ans_df = pd.concat([x, tes, pre], axis=1)
    ans_df = ans_df[['x', 'sin(label)', 'sin(prediction)', 'cos(label)', 'cos(prediction)']]
    ans_df.to_csv('../output/result_{}_{}_{}.csv'.format(str(i).zfill(2), len(hidden_layer_size), hidden_layer_size[0]))
    sin_mse = mean_squared_error(tes['sin(label)'].tolist(), pre['sin(prediction)'].tolist())
    cos_mse = mean_squared_error(tes['cos(label)'].tolist(), pre['cos(prediction)'].tolist())
    r2 = model.score(X_test, y_test)
    score_df.loc['{}'.format(i)] = [sin_mse, cos_mse, r2]

score_df.to_csv('../output/score.csv')
