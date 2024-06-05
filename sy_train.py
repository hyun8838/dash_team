import sqlite3
import pandas as pd
import numpy as np
import pickle as pkl

db_name = "my_db.db"
conn = sqlite3.connect(db_name)
c = conn.cursor()
c.execute("SELECT * FROM dat")
cols = [col[0] for col in c.description]
dat = pd.DataFrame(data=c.fetchall(),columns=cols)
conn.close()

## train, test split
from sklearn.model_selection import train_test_split
train, test = train_test_split(dat, test_size=0.3)
train.head()

## x variables preprocessing 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
X_cols = dat.loc[:, [i for i in list(dat.columns) if i not in ['재구매여부']]].columns
numeric_features = train[X_cols].select_dtypes(include=np.number).columns.tolist()
categorical_features = train[X_cols].select_dtypes(exclude=np.number).columns.tolist()
trans = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])
trans.fit(train[X_cols])
X_train = trans.transform(train[X_cols])
# Get feature names
num_features = numeric_features
cat_features = trans.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
all_features = list(num_features) + list(cat_features)
X_train = pd.DataFrame(X_train, columns=all_features)

## y variables preprocessing
y_train = train['재구매여부']

## classification modeling 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
model = LogisticRegression()
model.fit(X_train, y_train)

## save model and transformation 
print("creating model file...")
pkl.dump(model, open("model.pkl","wb"))
print("creating transformation file...")
pkl.dump(trans, open("transform.pkl","wb"))

## make prediction for testset
X_test = trans.transform(test[X_cols])
pred = model.predict(X_test)

## validation for testset
y_test = test['재구매여부']
print(np.mean(pred==y_test))
print(pd.crosstab(y_test, pred))
