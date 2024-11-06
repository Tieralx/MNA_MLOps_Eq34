from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle

data = pd.read_csv("sobar-72.csv")

# data (as pandas dataframes) 
X_train, X_test, y_train, y_test = train_test_split(data.drop('ca_cervix', axis=1), data['ca_cervix'], test_size = 0.2, random_state= 10)

model = LogisticRegression(solver = 'lbfgs',
                           C= 0.01,  # Regularization parameter for Logistic Regression
                           max_iter = 1000,  # Number of iterations
                           penalty = 'l2', 
                           random_state=4)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Model trained with an accuracy of: {accuracy:.2f}")

with open("cervical_cancer_model.pkl", "wb") as f:
    pickle.dump(model,f)
print("model saved as 'cervical_cancer_model.pkl'")
