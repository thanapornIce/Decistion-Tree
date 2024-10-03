import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load and preprocess dataset
df = pd.read_csv('D:\Predict Income\income-prediction\dataJ.csv')

# Check the actual column names in the dataset
print("Columns in the dataset:", df.columns)

# Handling missing values
df.replace("?", np.nan, inplace=True)
df.fillna(df.mode().iloc[0], inplace=True)

# Cleaning the column names (removing leading/trailing spaces)
df.columns = df.columns.str.strip()

# Debug step: print column names to verify they're correct
print("Cleaned column names:", df.columns)

# Merge similar categories
df.replace(['Divorced', 'Married-AF-spouse', 'Married-civ-spouse', 
            'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed'],
           ['divorced', 'married', 'married', 'married',
            'not married', 'not married', 'not married'], inplace=True)

# Encode categorical columns
category_col = ['Age', 'Working Class', 'Education', 'Marital Status','Occupation', 'Relationship','Gender', 'Capital Gain', 'Capital Loss', 'Hours per Week', 'Income'] 

# Ensure all the columns are present
for col in category_col:
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in the dataset.")

labelEncoder = preprocessing.LabelEncoder()
mapping_dict = {}
for col in category_col:
    df[col] = labelEncoder.fit_transform(df[col])
    mapping_dict[col] = dict(zip(labelEncoder.classes_, labelEncoder.transform(labelEncoder.classes_)))

# Split dataset for training
X = df.drop(columns=['Income']).values  # features
Y = df['Income'].values  # target

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)

# Train decision tree model
dt_clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=5, min_samples_leaf=5)
dt_clf_gini.fit(X_train, y_train)

# Predict and calculate accuracy
y_pred = dt_clf_gini.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Save the trained model
with open('model.pkl', 'wb') as f:
    pickle.dump(dt_clf_gini, f)

