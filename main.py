#Import Libraries for data manipulation and graph plot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Reading iris .csv data using pandas library
iris_dt = pd.read_csv('iris.data', sep=",", names = ["Sepal Length","Sepal Width", "Petal Length","Petal Width","Species"])

#getting head of the data to see the values and describing the dataset to understand data types
print(iris_dt.head())
print(f"\n{iris_dt.info()}")
print((f"\n{iris_dt.describe()}"))



print(f"\n{iris_dt['Species'].value_counts()}")

#datavisualisation
sns.set(rc={'figure.figsize':(5,5)})
gplot = sns.pairplot(iris_dt, hue='Species', markers='x')
gplot = gplot.map_upper(plt.scatter)
gplot = gplot.map_lower(sns.kdeplot)
sns.violinplot(x='Sepal Length', y='Species', data=iris_dt, inner='stick', palette='autumn')
plt.show()
sns.violinplot(x='Sepal Width', y='Species', data=iris_dt, inner='stick', palette='autumn')
plt.show()
sns.violinplot(x='Petal Length', y='Species', data=iris_dt, inner='stick', palette='autumn')
plt.show()
sns.violinplot(x='Petal Width', y='Species', data=iris_dt, inner='stick', palette='autumn')
plt.show()

itr=10000
a=0.01 #learning rate
arr=np.zeros(itr)

#function to assign values to class label by taking dataset as input
def mylabeltonum(iris_data):
    n_values = {'Iris-setosa': 1, 'Iris-versicolor': 2, 'Iris-virginica': 3}
    x_t = iris_data.drop(['Species'], axis=1)
    x_t['Bias'] = 1
    x_t= x_t.values
    y_t = iris_data['Species'].replace(n_values)
    y_t = y_t.values.reshape(x_t.shape[0], 1)
    return x_t, y_t

x,y=mylabeltonum(iris_dt)


#Model training for beta value computation
def mylinear_regression(x,y):
    np.random.seed(0)
    beta_value=np.random.randn(1,5)
    print("Beta:",beta_value)
    for i in range(itr):
        temp1=(np.dot(x,beta_value.T)-y)
        temp=np.sum(temp1 ** 2)
        arr[i]=(1/(2*x.shape[0])*temp)
        beta_value=beta_value-((a/x.shape[0])*np.dot((temp1).reshape(1,x.shape[0]),x))
    diag=plt.subplot(111)
    diag.plot(np.arange(itr),arr)
    plt.ylabel("Cost")
    plt.xlabel("Iterations")
    plt.title("RMS Error vs Alpha(0.01)")
    plt.show()
    return beta_value

#training the model and obtaining Beta Value

train_model=mylinear_regression(x,y)
predict=np.round(np.dot(x,train_model.T))
acc=(sum(predict==y)/float(len(y))*100)[0]
print("Accurcay of the model:",acc,predict) # accuracy and classification.

iris_dt['Species']=y
iris_dt['Bias']=1
iris_dt=iris_dt[["Bias","Sepal Length","Sepal Width", "Petal Length","Petal Width","Species"]]
iris_dt.head()

# cross-validation to compute using different split sizes for train and test data
def cv(iris_data, K=10):
    iris_data = iris_data.sample(frac=1)
    data = np.array_split(iris_data, K)
    acc = []
    for i in range(len(data)):
        test = data[i] # Choosing the 10% data splits as testing
        x_test = test.iloc[:,:-1].values
        y_test = test.iloc[:,-1].values # Y_test
        y_test = y_test.reshape(len(y_test), 1)
        #tmp = data[:]
        data[:].pop(i) # Choosing the other 90% data splits as training
        train_model = pd.concat(data[:]) #merge all the splits forming training DS
        x_train = train_model.iloc[:,:-1].values
        y_train = train_model.iloc[:,-1].values
        y_train = y_train.reshape(len(x_train),1)
        beta_value = mylinear_regression(x_train, y_train)

        # train function returns beta value
        predict = np.round(np.dot(x_test,beta_value.T))
        acc.append((sum(predict == y_test)/float(len(y_test)) * 100)[0])
    return acc

accuracy=cv(iris_dt,K=10)
cross_val_acc=sum(accuracy)/10
print("Cross Validation Accuracy:",cross_val_acc)