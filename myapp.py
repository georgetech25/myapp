import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score





st.sidebar.title('PAGES')
st.sidebar.image('sidebar.png')
menu = st.sidebar.radio('SELECT PAGE',('Convert String dataset to numeric','Train and Predict Model'))

if menu =='Convert String dataset to numeric':
    st.title('GEORGE AUTO ML ALGORITHYM')
    st.subheader('Convert String dataset to numeric')
    
    
    file_uploader = st.file_uploader('Upload your csv file Here', type=['csv'])
    
    if file_uploader:
        df = pd.read_csv(file_uploader)
        st.write(df.head())
        
        
        
        encoding_method = st.radio(
            "Select Encoding Method", 
            ("Label Encoding", "One-Hot Encoding")
        )

        if encoding_method == "Label Encoding":
            label_encoders = {}
            for column in df.select_dtypes(include=["object"]).columns:
                le = LabelEncoder()
                df[column] = le.fit_transform(df[column])
                label_encoders[column] = le
            st.write("### Label Encoded Data", df.head())

        elif encoding_method == "One-Hot Encoding":
            label_encoders = {}
            for column in df.select_dtypes(include=["object"]).columns:
                le = LabelEncoder()
                df[column] = le.fit_transform(df[column])
                label_encoders[column] = le
            st.write("### Label Encoded Data", df.head())
        
        
        
        
        # Download the encoded data
        st.download_button(
            label="Download Encoded CSV",
            data=df.to_csv(index=False),
            file_name="encoded_data.csv",
            mime="text/csv"
        )
               
                  
    st.image('download (1).jpeg',width=600)
    # st.divider()
    # st.divider()
    # st.divider()C:\Users\DELL\Desktop\bot\download (1).jpeg
        # EXCEL FILE UPLOAD
        
        
        
    file_uploader = st.file_uploader('Upload your excel file Here', type=['XLsx'])
    
    if file_uploader:
        df = pd.read_excel(file_uploader)
        st.write(df.head())
        
        
        
        encoder_method = st.radio(
            "Choose from the option", 
            ("LabelEncoding", "OneHotEncoding")
        )

        if encoder_method == "LabelEncoding":
            label_encoders = {}
            for column in df.select_dtypes(include=["object"]).columns:
                le = LabelEncoder()
                df[column] = le.fit_transform(df[column])
                label_encoders[column] = le
            st.write("### Label Encoded Data", df.head())

        elif encoder_method == "OneHotEncoding":
            label_encoders = {}
            for column in df.select_dtypes(include=["object"]).columns:
                le = LabelEncoder()
                df[column] = le.fit_transform(df[column])
                label_encoders[column] = le
            st.write("### Label Encoded Data", df.head())
        
        
        
        
        # Download the encoded data
        st.download_button(
            label="Download Encoded CSV",
            data=df.to_csv(index=False),
            file_name="encoded_data.csv",
            mime="text/csv"
        )
    
    
    
   
        # TRAIN MODEL( page 2)
        

elif menu == 'Train and Predict Model':
    st.title('MODEL TRAINGING')
    st.subheader('Train and predict your model')
    st.image('download.jpeg',width=700)
    
    
    
     
    file_uploader = st.file_uploader('Upload your  file Here', type=['csv'])
    
    if file_uploader:
        df = pd.read_csv(file_uploader)
        st.write(df.head())
        
        
        target_column = st.selectbox('select target column',df.columns)
        
        if st.button('Train Model'):
            X = df.drop(target_column,axis =1)
            y = df[target_column]
            X_train,y_train,X_test,y_test = train_test_split(X,y,test_size = 0.3, random_state = 42)
            model = DecisionTreeClassifier()
            model.fit(X,y)
            y_pred = model.predict(X)
            accuracy = accuracy_score(y,y_pred)*100
            st.write(f'Model accuracy score :{accuracy:.2f}%')
            prediction_array= np.array([y])
            st.write(prediction_array)
            
            
            
            
            
       
       
        
        
        
        
       
               
                  
        
    # st.divider()
    # st.divider()
    # st.divider()
        

        # EXCEL FILE UPLOAD
        
        
        
    # file_uploader = st.file_uploader('Upload your transformed file Here', type=['csv'])
    
    # if file_uploader:
    #     df = pd.read_csv(file_uploader)
    #     st.write(df.head())
    
        
        
    #     target_column = st.selectbox('select target column',df.columns)
        
    #     if st.button('Train Model'):
    #         X = df.drop(target_column,axis =1)
    #         y = df[target_column]
    #         X_train,y_train,X_test,y_test = train_test_split(X,y,test_size = 0.3, random_state = 42)
    #         model = DecisionTreeClassifier()
    #         model.fit(X,y)
    #         y_pred = model.predict(X)
            
    #         accuracy = accuracy_score(y,y_pred)*100
    #         st.write(f'Model accuracy score :{accuracy:.2f}%')
    #         prediction_array= np.array([y])
    #         st.write(prediction_array)
            
        
       
        

            
        
        
        
        
    