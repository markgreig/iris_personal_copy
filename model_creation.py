from sklearn.datasets import load_iris # used to get data
from sklearn.ensemble import RandomForestClassifier # classifier model used
from sklearn.model_selection import train_test_split # used to test on data not used to train model
from sklearn.metrics import accuracy_score # used to evaluate model
import pickle # used to save the model

data = load_iris()

# create instance of random forest class
model = RandomForestClassifier(random_state=1)

# train-test split the data
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)

# train the model on the training data
model = model.fit(X_train, y_train)

# test the model on the testing data
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Model accuracy is {round(accuracy*100, 2)}%')

# pickle the model
filename = 'model.pickle'
pickle.dump(model, open(filename, 'wb'))
