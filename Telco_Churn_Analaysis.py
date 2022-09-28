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