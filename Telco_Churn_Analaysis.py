import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv("D:\Windows10\Dekstop\Telco_Chustomer_Churn_Feature_Engineering/Telco-Customer-Churn.csv")

df.head()

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    :param dataframe: Telco company customers datasets
    :param cat_th: categoric variables thresholds value
    :param car_th: cardinal variables thresholds value
    :return: categoric, numeric and cardinal variables separeted
    """

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == 'O']
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != 'O']

    cat_but_car = [col for col in cat_cols if dataframe[col].nunique() > cat_th]
    num_but_cat = [col for col in num_cols if dataframe[col].nunique() < car_th]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    cat_cols += num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

##Numerical Variables Analysis

def num_summary(dataframe,numerical):
    quartiles = [0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    return dataframe[numerical].describe(quartiles)

print(num_summary(df,'tenure'))
for val in num_cols:
    print(num_summary(df,val))

##Categorical Variables Analysis

def cat_summary(dataframe, categorical):
    print(pd.DataFrame({categorical: dataframe[categorical].value_counts(),
                  'Ratio': 100 * dataframe[categorical].value_counts() / len(dataframe)}))
    print("\n")

for col in cat_cols:
    cat_summary(df, col)
df.head()
#Catching outliers:
q1  = df["MonthlyCharges"].quantile(0.25)
q3 = df["MonthlyCharges"].quantile(0.75)
iqr = q3 - q1
low = q1 - iqr * 1.5
up = q3 + iqr * 1.5
df[(df["MonthlyCharges"] < low) | (df["MonthlyCharges"] > up)]
#İdentifiying outliers threshold values with outlier_thresholds function
def outlier_thresholds(dataframe, col_name, q1=0.25, q3 = 0.75):
    """

    :param dataframe:Telco_Customer_churn dataframe
    :param col_name: desired column to spesify outliers
    :param q1:threshold value for lower values
    :param q3:threshold value for higher values
    :return: return the identified limits(thresholds)
    """
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    iqr = quartile3 - quartile1
    low_limit = quartile1 - iqr * 1.5
    up_limit = quartile3 + iqr * 1.5
    return low_limit, up_limit


#Checking columns with using outlier thresholds function
def check_outlier(dataframe, col_name):
    low, up = outlier_thresholds(df, col_name)

    if dataframe[(dataframe[col_name] < low) | (dataframe[col_name] > up)].any(axis=None):
        return True
    else:
        return False


for col in num_cols:
    print(col, check_outlier(df,col))
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    df.loc[(dataframe[variable] < low_limit), variable] = low_limit
    df.loc[(dataframe[col] > up_limit), variable] = up_limit

for col in num_cols:
    print(col, replace_with_thresholds(df, col))

df.isnull().sum()
