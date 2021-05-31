
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,f1_score, roc_auc_score,confusion_matrix,auc,roc_curve
import graphviz 
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

train = pd.read_csv('train_cc.csv')
test = pd.read_csv('test_cc.csv')


categorical_columns = ['Gender','Region_Code','Occupation','Channel_Code','Credit_Product','Is_Active']
numerical_columns = ['Age','Vintage','Avg_Account_Balance']

def categorical_plots(train_df,categorical_columns):
    fig,axes = plt.subplots(3,2,figsize=(12,15))
    for idx,cat_col in enumerate(categorical_columns):
        row,col = idx//2,idx%2
        sns.countplot(x=cat_col,data=train_df,hue='Is_Lead',ax=axes[row,col])
    plt.subplots_adjust(hspace=1)

def numerical_plots(train_df, numerical_columns):
    fig,axes = plt.subplots(1,3,figsize=(17,5))
    for idx,cat_col in enumerate(numerical_columns):
        sns.boxplot(y=cat_col,data=train_df,x='Is_Lead',ax=axes[idx])
    print(train_df[numerical_columns].describe())
    print(40*"*")
    plt.subplots_adjust(hspace=1)
def corrMap(train_df):
    print('Correlation Values')
    print(train_df.corr())
    print(40*"*")

def eda(train_df, categorical_columns, numerical_columns):
    categorical_plots(train_df, categorical_columns)
    numerical_plots(train_df, numerical_columns)
    corrMap(train_df)

train_data = train.drop(columns = ['ID'])

def balancing(train_df):
    lead_count_0, lead_count_1 = train_df['Is_Lead'].value_counts()
    lead_0 = train_df[train_df['Is_Lead'] == 0]
    lead_1 = train_df[train_df['Is_Lead'] == 1]
    lead_1_over =  lead_1.sample(lead_count_0, replace=True)
    balanced_df = pd.concat([lead_1_over, lead_0], axis=0)
    print(balanced_df['Is_Lead'].value_counts())
    return balanced_df
    #balanced_df['Is_Lead'].value_counts()
def scaling(train_df, numerical_columns):
    for i in numerical_columns:
        scale = StandardScaler().fit(train_df[[i]])
        train_df[i] = scale.transform(train_df[[i]])
    return train_df

def encoding(train_df):
    train_df_encoded = pd.get_dummies(train_df,drop_first=True)
    return train_df_encoded
def impute_nan_create_category(DataFrame,ColName):
    DataFrame[ColName] = np.where(DataFrame[ColName].isnull(),"Unknown",DataFrame[ColName])
    return DataFrame
def imputing(train_df):
    #imp = SimpleImputer(strategy="most_frequent")
    #imp_train = imp.fit(train_df)
    #train_df = imp_train.transform(train_df)
    train_df = impute_nan_create_category(train_df,'Credit_Product')
    return train_df


train_impute = imputing(train_data)
train_balance = balancing(train_impute)
X_df = encoding(train_balance)
X = X_df.drop(columns='Is_Lead')
y = X_df['Is_Lead']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state =42)
X_train1 = X_train.copy()
X_train_scaled = scaling(X_train1,numerical_columns)
print(X_train.head(10))
print(X_train_scaled.head(10))
X_test1 = X_test.copy()
X_test_scaled = scaling(X_test1,numerical_columns)

X_1 = X.copy()
X_scaled = scaling(X_1,numerical_columns)

def get_test_data():
    test_data = test.drop(columns = ['ID'])
    test_impute = imputing(test_data)
    test_df = encoding(test_impute)
    test_df1 = test_df.copy()
    test_scaled = scaling(test_df1,numerical_columns)
    return test_df,test_scaled

test_orig, test_scale = get_test_data()
model = RandomForestClassifier(random_state = 1,n_estimators=400,max_depth = 15,max_features=10,min_samples_leaf = 2)
model.fit(X,y)
y_predicted = model.predict(test_orig)
final_data = {'ID': test.ID, 'Is_Lead': y_predicted}
rf_result = pd.DataFrame(data=final_data)
print(rf_result['Is_Lead'].value_counts())
rf_result.to_csv('randomForest_result.csv',index=False)