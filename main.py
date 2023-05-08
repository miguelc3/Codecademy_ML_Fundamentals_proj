import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


path_excel = r'C:\Users\Miguel\pyproj\Codecademy_ML_fundamentals_proj\DryBeanDataset\Dry_Bean_Dataset.xlsx'
df = pd.read_excel(path_excel)

classes = df['Class'].unique().tolist()
dict_conversion = {}
features = df.columns.tolist()[:-1]

# Separate features from classification
X = df[features]
y = df['Class']


# Apply min max scaller to all features -> so that all have the same wheight +-
scaler = MinMaxScaler()

# Fit scaler to features
scaler.fit(X)

# Transform features
X_scaled = scaler.transform(X)

# Split to train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# Logistic Regression
# Create model
model = LogisticRegression(random_state=42, max_iter=10000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accracy = accuracy_score(y_test, y_pred)


print('Logistic Regression accuracy: ', accracy)

# Decision Tree
# Create model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accracy = accuracy_score(y_test, y_pred)

print('Decision Tree accuracy: ', accracy)


# Random Forest
# Create model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accracy = accuracy_score(y_test, y_pred)
print('Random Forest accuracy: ', accracy)

# Support Vector Machines
# Create model
model = SVC(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accracy = accuracy_score(y_test, y_pred)

print('Support Vector Machines accuracy: ', accracy)

