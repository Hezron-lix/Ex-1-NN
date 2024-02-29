<H3>NAME: P HEZRON BELIX</H3>
<H3>REGISTER NO.: 212223230078</H3>
<H3>EX. NO.1</H3>
<H3>DATE</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:

```python
#import libraries
from google.colab import files
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#Read the dataset from drive
df=pd.read_csv('/content/Churn_Modelling.csv')
df

#Finding Missing Values
print(df.isnull().sum())

#Handling Missing values
df.fillna(df.mean(),inplace=True)
print(df.isnull().sum())

y=df.iloc[:,-1].values
print(y)

#Check for Duplicates
df.duplicated()

#Detect Outliers
df.describe()

#Normalize the dataset
scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)

#split the dataset into input and output
x=df.iloc[:, :-1].values
print(x)
y=df.iloc[:,-1].values
print(y)

#splitting the data for training & Testing
X_train ,X_test ,y_train,y_test=train_test_split(x,y,test_size=0.2)

#Print the training data and testing data
print("X_train\n")
print(X_train)
print("\nLenght of X_train ",len(X_train))
print("\nX_test\n")
print(X_test)
print("\nLenght of X_test ",len(X_test))
```

## OUTPUT:

### dataset:

![img1](https://github.com/Hezron-lix/Ex-1-NN/assets/139331337/b2cb1846-2247-4c54-9b67-e5b18762496f)


### Finding Missing Values:

![image-2](https://github.com/Hezron-lix/Ex-1-NN/assets/139331337/1cf59ecb-02b7-4dce-940d-2b2692b18a37)


### Handling Missing values:

![image-3](https://github.com/Hezron-lix/Ex-1-NN/assets/139331337/ef033dda-e334-42df-ba56-cbcccc43eba7)


### Duplicates:

![image-4](https://github.com/Hezron-lix/Ex-1-NN/assets/139331337/e225c32c-5b69-4a9e-832f-80bb0a5d37ab)


### Normalize the dataset:

![image-5](https://github.com/Hezron-lix/Ex-1-NN/assets/139331337/738c0cf3-f49c-4522-92df-92b4e9ea4d40)


### split the dataset into input and output:

![image-6](https://github.com/Hezron-lix/Ex-1-NN/assets/139331337/a5c41377-434e-43c5-b2d8-1aca3f451184)


### splitting the data for training & Testing:

![image-7](https://github.com/Hezron-lix/Ex-1-NN/assets/139331337/532e72a7-7eb5-401e-8efa-d8f5140d3ac5)


## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


