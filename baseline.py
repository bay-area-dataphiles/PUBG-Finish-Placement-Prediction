import pandas as pd
from sklearn.linear_model import LinearRegression

train = pd.read_csv('../input/train_V2.csv')
test = pd.read_csv('../input/test_V2.csv')

X = train
X = X.select_dtypes(include=['int64', 'float64'])
X.dropna(inplace=True)
y = X.pop('winPlacePerc')

X_test = test.select_dtypes(include=['int64', 'float64'])

clf = LinearRegression()
clf.fit(X, y)

submission = clf.predict(X_test)
pd.DataFrame(submission, columns=['winPlacePerc']).set_index(test.Id).to_csv('submission.csv')
