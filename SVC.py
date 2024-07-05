#USING SVC MODEL

#imports
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC

df = pd.read_csv("loan_status.csv")
print(df.head())
print(df.shape)

# #dataset info
# print(df.isnull().sum())
# print(df.dtypes)
# print(df.describe())

#Handling Null values in categorical cols-> drop them using .dropna()
clean_df = df.dropna()
print(clean_df.shape) 

# print(clean_df.isnull().sum()) #check for null values after dropping some data

#Handling Categorical Values
"""replace data with either 0 or 1 """
clean_df.replace({"Gender":{"Male":1, "Female":0}}, inplace=True)
clean_df.replace({"Married":{"Yes":1, "No":0}}, inplace=True)
clean_df.replace({"Education":{"Graduate":1, "Not Graduate":0}}, inplace=True)
clean_df.replace({"Self_Employed":{"Yes":1, "No":0}}, inplace=True)
clean_df.replace({"Property_Area":{"Rural":0, "Semiurban":1, "Urban":2}}, inplace=True)
clean_df.replace({"Loan_Status":{"Y":1, "N":0}}, inplace=True)
clean_df.replace(to_replace='3+', value=3, inplace=True)
clean_df['Dependents'] = clean_df['Dependents'].astype(int)

#drop Loan_ID col
clean_df = clean_df.drop(columns='Loan_ID')

#checking the processed dataset
print(clean_df.head())
print(clean_df.dtypes)


#Data Analysis
#corelation between features and label
sns.heatmap(clean_df.corr(), annot=True)
plt.title('Corelation heatmap')
plt.show()


#Let’s also see if there is any bias between “Male” and “Female”, “Married” and “Unmarried”, and “Graduate” and “Not Graduate” in giving loan
#gender(male =1 ,female = 0)
sns.countplot(data=clean_df, x = 'Gender', hue='Loan_Status')
plt.show()

#married = 1, unmarried = 0
sns.countplot(data=clean_df, x = 'Married', hue='Loan_Status')
plt.show()

#grad = 1, Nograd = 0
sns.countplot(data=clean_df, x = 'Education', hue='Loan_Status')
plt.show()


#Splitting the Dataset
y = clean_df['Loan_Status']
X = clean_df.drop(columns='Loan_Status')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=234)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

#Building Model

svc = SVC(kernel='linear')
svc.fit(X_train, y_train)
svc_prediction = svc.predict(X_test)

#accuracy
svc_accuracy = accuracy_score(y_test, svc_prediction)
print(svc_accuracy)

#confusion matrix
confmat = confusion_matrix(y_test, svc_prediction)
print(confmat)

sns.heatmap(confmat, annot=True)
plt.title('confusion matrix')
plt.show()