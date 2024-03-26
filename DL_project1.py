import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import InputLayer,Dense

#Giving datasets

# Function to create and train a simple neural network
def train_neural_network(x_train, y_train, num_epochs,learning_rate,act_fun,batch_size,x_test,y_test):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation=act_fun, input_shape=(2,)),
        tf.keras.layers.Dense(4, activation=act_fun),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    optimizer1=tf.keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(optimizer=optimizer1, loss='binary_crossentropy', metrics=['accuracy'])

    #history=model.fit(x_train,y_train,epochs=num_epochs,batch_size=batch_size,validation_split=0.2)
    history=model.fit(x_train,y_train,epochs=num_epochs,batch_size=batch_size,validation_data=(x_test,y_test))
    return model,history
def plot_decision_regions(model, X, y):
    h=0.2 # step size in the mesh

    # Create a meshgrid of feature values
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Predict the class probabilities for each point in the meshgrid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Reshape the predictions to match the meshgrid shape
    Z = Z.reshape(xx.shape)


    # Plot the decision regions
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z,  alpha=0.8)

    # Plot the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Regions')
    plt.show()
    st.pyplot(plt)


# Main function
def main():
    st.title('TensorFlow Playground Clone')
    st.sidebar.header('Settings')
    
    datasets= {
    "Concentriccir1" :r"C:\Users\LENOVO\Documents\today\Multiple CSV\2.concerticcir1.csv",
    "Ushape":r"C:\Users\LENOVO\Documents\today\Multiple CSV\1.ushape.csv",
    "Concentriccir2":r"C:\Users\LENOVO\Documents\today\Multiple CSV\3.concertriccir2.csv",
    "Linearsep":r"C:\Users\LENOVO\Documents\today\Multiple CSV\4.linearsep.csv",
    "Outlier":r"C:\Users\LENOVO\Documents\today\Multiple CSV\5.outlier.csv",
    "Overlap":r"C:\Users\LENOVO\Documents\today\Multiple CSV\6.overlap.csv",
    "XOR":r"C:\Users\LENOVO\Documents\today\Multiple CSV\7.xor.csv",
    "Spirals":r"C:\Users\LENOVO\Documents\today\Multiple CSV\8.twospirals.csv",
    "Random":r"C:\Users\LENOVO\Documents\today\Multiple CSV\9.random.csv",
    
    }
    
    dataset1 = st.sidebar.selectbox('Select your dataset',list(datasets.keys()))
    #if dataset1=="upload_file":
    ##    data=pd.read_csv(upload_file1)
    #else:                      
    d=datasets[dataset1]
    data=pd.read_csv(d)
    num_epochs = st.sidebar.slider('Number of Epochs', min_value=10, max_value=1000, value=15)
    act_fun=st.sidebar.selectbox("Select activation",["sigmoid","relu","tanh","softmax"])
    learning_rate = st.sidebar.selectbox("Select Learning rate ",[0.0001,0.001,0.005,0.002,0.01,0.1,0.5,1.0,2.0,3.0,7.0])
    batch_size=st.sidebar.slider("Select batch size " ,min_value=10,max_value=40)

    fv=np.array(data.iloc[:,0:-1])
    Y = data.iloc[:,-1]
    if Y.dtype != int:
        Y= Y.astype(int)

    cv=np.array(Y)

    x_train,x_test,y_train,y_test=train_test_split(fv,cv,test_size=0.2,random_state=12,stratify=cv)
    #x_train,x_cv,y_train,y_cv=train_test_split(x_train,y_train,test_size=0.2,random_state=12,stratify=y_train)



    if st.sidebar.button("Submit"):

        st.subheader('Training')
        model,history = train_neural_network(x_train, y_train, num_epochs,learning_rate,act_fun,batch_size,x_test,y_test)
        st.write('Model trained successfully!')
        
        st.subheader('Loss Curves')
        st.line_chart(pd.DataFrame({'train_loss': history.history['loss'], 'val_loss': history.history['val_loss']}))

        st.subheader('Decision Boundary')
    
        plot_decision_regions(model,x_test, y_test)


if __name__ == '__main__':
    main()
