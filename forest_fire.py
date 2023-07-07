# #!C:\Users\Lenovo\AppData\Local\Programs\Python\Python37-32\python.exe

# import numpy as np
# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# import warnings
# import pickle
# data = pd.read_csv("data.csv")
# print(data.head(5))
# data = np.array(data)
# X = data[1:, 1:-1]
# y = data[1:, -1]
# y = y.astype('int')
# X = X.astype('int')
# # print(X,y)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# log_reg = LogisticRegression()


# log_reg.fit(X_train, y_train)

# # inputt=[int(x) for x in "45 32 60 40".split(' ')]
# # final=[np.array(inputt)]

# #b = log_reg.predict_proba(final) # type: ignore


# pickle.dump(log_reg,open('model.pkl','wb'))
# model=pickle.load(open('model.pkl','rb'))

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load the csv file
#return render_template("index.html", prediction_text = "The Sepsis is {}".format(prediction))
df = pd.read_csv("data.csv")

print(df.head())

# Select independent and dependent variablesource sess
X = df[["HR", "O2Sat", "SBP", "MAP","Resp"]]
y = df["SepsisLabel"]

# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test= sc.transform(X_test)

# Instantiate the model
classifier = RandomForestClassifier()

# Fit the model
classifier.fit(X_train, y_train)

# Make pickle file of our model
pickle.dump(classifier, open("model.pkl", "wb"))
