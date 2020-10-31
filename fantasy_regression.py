import numpy as np
import pandas as pd
pd.set_option('max_columns', None)
pd.set_option('display.max_rows', None)
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
sc = StandardScaler()

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor

df = pd.read_csv('QBdata.csv')

for i in df:
    if (i != 'Player') and (i != 'Draft') and (i != 'Tm') and (i != 'Lg'):
        df[i] = pd.to_numeric(df[i], errors='coerce')
        
df_2018 = df.loc[df['Year'] == 2018]
df_2018 = df_2018.dropna()
df_2018_adjusted = sc.fit_transform(df_2018[['Age', 'G', 'Cmp', 'Att', 'Yds', 'TD', 'Int', 'Att.1', 'Yds.1', 'TD.1', 'Rec', 'Yds.2', 'TD.2', 'Fmb']])

df = df.loc[df['Year'] != 2018]
size = df.shape[0]

df = df[['Player', 'PPR', 'Age', 'G', 'Cmp', 'Att', 'Yds', 'TD', 'Int', 'Att.1', 'Yds.1', 'TD.1', 'Rec', 'Yds.2', 'TD.2', 'Fmb']]
df = df.sort_values(['Player', 'Age'], ascending = [True, False])
grouped = df.groupby(df.Player)
frames = []
for name, group in grouped:
    group.sort_values(['Age'], ascending = [False])
    group = group.shift(1)
    frames.append(group)
df = pd.concat(frames)
df = df.dropna()

X = df[['Age', 'G', 'Cmp', 'Att', 'Yds', 'TD', 'Int', 'Att.1', 'Yds.1', 'TD.1', 'Rec', 'Yds.2', 'TD.2', 'Fmb']]
X = sc.fit_transform(X)

df['PPR'] = df['PPR'].fillna(1)
Y = df['PPR']
Y = np.asanyarray(Y)
Y = Y.reshape(-1, 1)
Y[np.isnan(Y)] = 0

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.1)

model = Sequential()
model.add(Dense(14, input_dim = 14, activation = 'relu'))
model.add(Dense(14, activation = 'relu'))
model.add(Dense(14, activation = 'relu'))
model.add(Dense(14, activation = 'relu'))
model.add(Dense(1, activation = 'linear'))
model.compile(optimizer='Adam', loss='mse')


history = model.fit(X, Y, epochs = 250, validation_split=0.2)

predictions = model.predict(df_2018_adjusted[:])
df_results = df_2018[['Player']]
df_results['Predictions'] = predictions
df_results = df_results.sort_values(['Predictions'], ascending = [False])


actual_dict = {'Patrick Mahomes': 292.04, 'Ben Roethlisberger': 13.74, 'Aaron Rodgers': 282.38, 'Andrew Luck': 0, 'Deshaun Watson': 331.98, 'Drew Brees': 228.76, 'Jared Goff': 263.52, 'Russell Wilson': 333.60, 'Kirk Cousins': 250.42, 'Tom Brady': 271.68, 'Dak Prescott': 348.78, 'Philip Rivers': 255.50, 'Cam Newton': 17.68, 'Matt Ryan': 281.34, 'Mitchell Trubisky': 212.82, 'Eli Manning': 59.38, 'Baker Mayfield': 250.18, 'Derek Carr': 252.36, 'Matthew Stafford': 171.56, 'Josh Allen': 297.56, 'Carson Wentz': 282.86, 'Jameis Winston': 335.36, 'Blake Bortles': -0.78, 'Andy Dalton': 216.66, 'Sam Darnold': 202.16, 'Marcus Mariota': 87.02, 'Lamar Jackson': 421.68, 'Ryan Fitzpatrick': 254.46, 'Joe Flacco': 89.88, 'Alex Smith': 0, 'Josh Rosen': 22.98, 'Ryan Tannehill': 230.18, 'Jeff Driskel': 61.10, 'Blaine Gabbert': 0, 'Joe Webb': 0, 'Terrelle Pryor': 0}
actual_values = list(actual_dict.values())
actual_values.sort()

actual = np.array(['Lamar Jackson', 'Dak Prescott', 'Jameis Winston', 'Russell Wilson', 'Deshaun Watson', 'Josh Allen', 'Kyler Murray', 'Patrick Mahomes', 'Carson Wentz', 'Aaron Rodgers', 'Matt Ryan', 'Tom Brady', 'Jared Goff', 'Jimmy Garoppolo', 'Philip Rivers', 'Ryan Fitzpatrick', 'Derek Carr', 'Kirk Cousins', 'Baker Mayfield', 'Gardner Minshew II', 'Ryan Tannehill', 'Drew Brees', 'Daniel Jones', 'Jacoby Brissett', 'Andy Dalton', 'Mitch Trubisky', 'Sam Darnold', 'Kyle Allen', 'Matthew Stafford', 'Mason Rudolph'])
espn_berry = np.array(['Patrick Mahomes', 'Deshaun Watson', 'Aaron Rodgers', 'Baker Mayfield', 'Carson Wentz', 'Matt Ryan', 'Kyler Murray', 'Ben Roethlisberger', 'Cam Newton', 'Jared Goff', 'Tom Brady', 'Drew Brees', 'Jameis Winston', 'Russell Wilson', 'Mitchell Trubisky', 'Lamar Jackson', 'Dak Prescott', 'Philip Rivers', 'Josh Allen', 'Kirk Cousins', 'Sam Darnold', 'Jimmy Garoppolo', 'Jacoby Brissett', 'Derek Carr', 'Matthew Stafford', 'Andy Dalton', 'Nick Foles', 'Ryan Fitzpatrick', 'Marcus Mariota', 'Dwayne Haskins'])
espn_clay = np.array(['Patrick Mahomes', 'Deshaun Watson', 'Matt Ryan', 'Aaron Rodgers', 'Cam Newton', 'Baker Mayfield', 'Carson Wentz', 'Dak Prescott', 'Tom Brady', 'Ben Roethlisberger', 'Drew Brees', 'Russell Wilson', 'Kyler Murray', 'Jared Goff', 'Lamar Jackson', 'Mitchell Trubisky', 'Jameis Winston', 'Philip Rivers', 'Kirk Cousins', 'Sam Darnold', 'Josh Allen', 'Derek Carr', 'Jimmy Garoppolo', 'Matthew Stafford', 'Marcus Mariota', 'Jacoby Brissett', 'Andy Dalton', 'Eli Manning', 'Joe Flacco', 'Nick Foles'])
espn_cockcroft = np.array(['Patrick Mahomes', 'Deshaun Watson', 'Aaron Rodgers', 'Matt Ryan', 'Baker Mayfield', 'Cam Newton', 'Carson Wentz', 'Russell Wilson', 'Drew Brees', 'Jared Goff', 'Jameis Winston', 'Ben Roethlisberger', 'Kyler Murray', 'Dak Prescott', 'Lamar Jackson', 'Tom Brady', 'Philip Rivers', 'Mitchell Trubisky', 'Kirk Cousins', 'Josh Allen',  'Jimmy Garoppolo', 'Sam Darnold', 'Derek Carr', 'Matthew Stafford', 'Andy Dalton', 'Jacoby Brissett', 'Nick Foles', 'Marcus Mariota', 'Joe Flacco', 'Eli Manning'])
espn_karabell = np.array(['Patrick Mahomes', 'Deshaun Watson', 'Aaron Rodgers', 'Matt Ryan', 'Carson Wentz', 'Cam Newton', 'Baker Mayfield', 'Kyler Murray', 'Russell Wilson', 'Dak Prescott', 'Ben Roethlisberger', 'Drew Brees', 'Tom Brady', 'Lamar Jackson', 'Jared Goff', 'Jameis Winston', 'Philip Rivers', 'Kirk Cousins', 'Josh Allen', 'Mitchell Trubisky', 'Sam Darnold', 'Derek Carr', 'Jimmy Garoppolo', 'Jacoby Brissett', 'Dwayne Haskins', 'Nick Foles', 'Andy Dalton', 'Matthew Stafford', 'Marcus Mariota', 'Eli Manning'])
espn_yates = np.array(['Patrick Mahomes', 'Deshaun Watson', 'Aaron Rodgers', 'Matt Ryan', 'Cam Newton', 'Baker Mayfield', 'Carson Wentz', 'Kyler Murray', 'Tom Brady', 'Russell Wilson', 'Ben Roethlisberger', 'Lamar Jackson', 'Drew Brees', 'Jameis Winston',  'Jared Goff', 'Dak Prescott', 'Sam Darnold', 'Jimmy Garoppolo', 'Mitchell Trubisky', 'Philip Rivers', 'Derek Carr', 'Josh Allen', 'Kirk Cousins',  'Jacoby Brissett', 'Nick Foles', 'Dwayne Haskins', 'Matthew Stafford', 'Joe Flacco', 'Andy Dalton', 'Eli Manning', 'Ryan Fitzpatrick'])

players = df_results["Player"]
players = players.to_frame()
players = players.to_numpy()
players = players[0:30]
players = players.astype(str)
players = players.flatten()

for i in range(30):
    players[i] = players[i].split("\\")[0]
    print(players[i])
    
comparison = pd.DataFrame()
comparison.insert(0, "Actual", actual)
comparison.insert(1, "ESPN", espn_karabell)
comparison.insert(2, "Sid", players)

def getME(other):
    score = 0
    
    for i in actual:
        a = np.where(other == i)[0]
        b = np.where(actual == i)[0]
        
        try:
            score += abs(b - a)
        except:
            score += abs(29 - b)
    return score[0]

def getAccuracy(other):
    score = 0
    
    for i in actual:
        if(i not in other):
            score += actual_values[30]
        else:
            a = np.where(actual == i)[0]
            b = np.where(other == i)[0]
            score += abs(actual_values[a[0]] - actual_values[b[0]])
            
    return score
    

print("Sid Mean Error:", getME(players))
print("ESPN Mean Error:", getME(espn_clay))
print("ESPN Mean Error:", getME(espn_berry))
print("ESPN Mean Error:", getME(espn_cockcroft))
print("ESPN Mean Error:", getME(espn_karabell))
print("ESPN Mean Error:", getME(espn_yates))

print("Sid Accuracy:", getAccuracy(players))
print("ESPN Accuracy:", getAccuracy(espn_clay))
print("ESPN Accuracy:", getAccuracy(espn_berry))
print("ESPN Accuracy:", getAccuracy(espn_cockcroft))
print("ESPN Accuracy:", getAccuracy(espn_karabell))
print("ESPN Accuracy:", getAccuracy(espn_yates))
