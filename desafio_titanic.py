import numpy as np
import pandas as pd
import math

from sklearn.tree import DecisionTreeClassifier


def generate_csv(id_values, predict_result, filename):
    count = 0
    values = []
    for id_value in id_values:
        values.append((int(id_value), int(predict_result[count])))
        count += 1

    df = pd.DataFrame(values, columns=['PassengerId','Survived']) 
    df.to_csv(filename + ".csv", sep=',', columns=['PassengerId','Survived'], index=False)

#treino
df = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")


df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
df['Gender'] = df['Sex'].map({'female': 0, 'male':1}).astype(float)
df['Port'] = df['Embarked'].map({'C':1, 'S':2, 'Q':3}).astype(float)
df = df.drop(['Sex', 'Embarked'], axis=1)
cols = df.columns.tolist()
cols = [cols[1]] + cols[0:1] + cols[2:]
df = df[cols]
df = df.fillna(-1)


dataset = {
    'data': df.values[0:, 2:],
    'target': df.values[0:,0]
}

X_train = dataset['data']
y_train = dataset['target']


#teste



df_test = df_test.drop(['Name', 'Ticket', 'Cabin'], axis=1)
df_test['Gender'] = df_test['Sex'].map({'female': 0, 'male':1}).astype(int)
df_test['Port'] = df_test['Embarked'].map({'C':1, 'S':2, 'Q':3}).astype(int)
df_test = df_test.drop(['Sex', 'Embarked'], axis=1)
df_test = df_test.fillna(-1)

test_ids = df_test.values[0:, 0]
test_dataset = df_test.values[0:, 1:]

#arvore de decisão
dt = DecisionTreeClassifier()

dt.fit(X_train, y_train)

Y_pred = dt.predict(test_dataset)

generate_csv(test_ids, Y_pred, "reultado_titan")

acc_decision_tree = round(dt.score(X_train, y_train) * 100, 2)



print(round(acc_decision_tree,2,), "%")