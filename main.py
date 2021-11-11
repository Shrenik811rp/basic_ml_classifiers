import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA



def main():
    st.title("Machine learning models")

    dataset = ['Iris', 'Breast Cancer', 'Wine']
    classifier = ['KNN', 'SVM', 'Random Forest']

    dataset_name = st.sidebar.selectbox("Select Dataset",(dataset[0], dataset[1], dataset[2]))
    st.subheader(dataset_name + " Dataset")

    classifier_name = st.sidebar.selectbox("Select Classifier",(classifier[0],classifier[1],classifier[2]))
    
    st.markdown("Model used : "+classifier_name)

    parameter = add_parameter(classifier_name,classifier)

    classifier_used = get_classifier(classifier_name,classifier,parameter)

    dataset_detail(classifier_used,dataset_name,classifier_name)

    return


def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    elif dataset_name == "Wine":
        data = datasets.load_wine()

    x = data.data
    y = data.target

    return x,y


def dataset_detail(model,dataset_name,classif_name):

    x,y = get_dataset(dataset_name)

    st.write("Shape of Dataset: ",x.shape)
    st.write("Classes in each dataset:",len(np.unique(y)))

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=10)
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test,y_pred)
    st.markdown(f"Accuracy score for {classif_name}:\t```{round((accuracy*100),2)}%```")

    plot_graph(x,y,dataset_name,model)

    return
    



def add_parameter(classif_name,classifier):
    params = dict()

    if classif_name == classifier[0]:
        k_nn = st.sidebar.slider("Nos Neighbors (k)",min_value=1,max_value=15,value=3)
        params["k_nn"] = k_nn
    elif classif_name == classifier[1]:
        c_svm = st.sidebar.slider("Regularization parameter (C) default = 1.0",min_value=0.01,max_value=10.0,value=1.0)
        params["c_svm"] = c_svm
    
    elif classif_name == classifier[2]:
        max_depth = st.sidebar.slider("max_depth default = None or 2",min_value=0,max_value=15,value=2)
        n_estim = st.sidebar.slider("n_estimators default = 100",min_value=1,max_value=1000,value=100)

        params['max_depth'] = max_depth
        params['n_estim'] = n_estim

    return params


def get_classifier(classif_name,classifier,params):
    
    if classif_name == classifier[0]:
        classifier_model = KNeighborsClassifier(n_neighbors=params["k_nn"])
    elif classif_name == classifier[1]:
        classifier_model = SVC(C=params['c_svm'])
    
    elif classif_name == classifier[2]:
        classifier_model = RandomForestClassifier(n_estimators=params['n_estim'],max_depth=params['max_depth'],random_state=10)
    return classifier_model


def plot_graph(x,y,dataset_name,model):

    pca = PCA(2)
    X_projected = pca.fit_transform(x)
    
    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]
    
    fig = plt.figure()
    plt.scatter(x1, x2,
        c=y, alpha=0.8,
        cmap='viridis')
    plt.title(f'{dataset_name} - {model}')    
    plt.xlabel('Component - 1')
    plt.ylabel('Component - 2')
    plt.colorbar()
    
    plt.show()
    st.pyplot(fig)

    return




if __name__ == "__main__":
    main()