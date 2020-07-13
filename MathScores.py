import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error

df = pd.read_csv('C:/Users/Filip/Code/DeepLearning/Datasets/StudentsPerformance.csv')

pd.options.display.max_columns = None

females = df[df.gender=='female']
males = df[df.gender=='male']

noLunch = df[df.lunch=='standard']
lunch = df[df.lunch=='free/reduced']

bachelors = df[df['parental level of education']=="bachelor's degree"]
some_college = df[df['parental level of education']=='some college']
masters = df[df['parental level of education']=="master's degree"]
associates = df[df['parental level of education']=="associate's degree"]
high_school = df[df['parental level of education']== 'high school']
some_high_school = df[df['parental level of education']=='some high school']


df2 = df.copy()

df2['gender'] = df2['gender'].replace({'male':1,'female':0})
df2['lunch'] = df2['lunch'].replace({'standard':1,'free/reduced':0})
df2['race/ethnicity'] = df2['race/ethnicity'].replace({'group A':1,'group B':2,'group C':3,'group D':4,'group E':5})
df2['parental level of education'] = df2['parental level of education'].replace({'some high school':1,'high school':2,'some college':3,"associate's degree":4,"bachelor's degree":5,"master's degree":6})
df2['test preparation course'] = df2['test preparation course'].replace({'completed':1,'none':0})



#print(df['race/ethnicity'].unique())
#print(df['parental level of education'].unique())
#print(df['test preparation course'].unique())

print(df['math score'].corr(df['reading score']))
print(df['math score'].corr(df['writing score']))
print(df['writing score'].corr(df['reading score']))



df2_prescaled = df2.copy()
df2_scaled = df2.copy()
df2_scaled = df2_scaled.drop(['math score'], axis=1)
df2_scaled = df2_scaled.drop(['reading score'], axis=1)
df2_scaled = df2_scaled.drop(['writing score'], axis=1)
df2_scaled = scale(df2_scaled)


cols = df2.columns.tolist()
cols.remove('math score')
cols.remove('reading score')
cols.remove('writing score')
df2_scaled = pd.DataFrame(df2_scaled, columns=cols, index=df2.index)
df2_scaled = pd.concat([df2_scaled, df2['math score']], axis=1)

df2 = df2_scaled.copy()

X = df2.loc[:,df2.columns!='math score']
y = df2.loc[:,'math score']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Sequential()
model.add(Dense(16, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(X_train, y_train, epochs=100)



def predict():
    sample = X_test.sample(n=1, random_state=np.random.randint(low=0, high=1000))
    print(sample)
    idx = sample.index[0]
    actual_score = df2_prescaled.loc[idx,'math score']
    gender = df2_prescaled.loc[idx,'gender']
    parEdu = df2_prescaled.loc[idx,'parental level of education']

    predicted_score = model.predict(sample)[0][0]
    print('gender: ', gender)
    print('parEdu: ', parEdu)
    print("Actual score: {:0.2f}".format(actual_score))
    print("Predicted score: {:0.2f}".format(predicted_score))

predict()

train_pred = model.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
test_pred = model.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
print("Train RMSE: {:0.2f}".format(train_rmse))
print("Test RMSE: {:0.2f}".format(test_rmse))

def visualize(dataList):
    bins = np.linspace(0, 100, 100) #min, max, num of steps
    i = 0
    for data in dataList:
        i=i+1
        plt.hist(data['math score'], bins, alpha=0.5, label = i)
    plt.legend(loc='upper right')
    plt.title('Math scores')
    plt.show()

    i = 0
    for data in dataList:
        i=i+1
        plt.hist(data['reading score'], bins, alpha=0.5, label = i)
    plt.legend(loc='upper right')
    plt.title('Reading scores')
    plt.show()
    
    i = 0
    for data in dataList:
        i=i+1
        plt.hist(data['writing score'], bins, alpha=0.5, label = i)
    plt.legend(loc='upper right')
    plt.title('Writing score')
    plt.show()

parentsEdu = [bachelors,some_college,masters,associates,high_school,some_high_school]
#visualize([masters, high_school])