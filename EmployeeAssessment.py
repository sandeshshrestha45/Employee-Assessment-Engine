# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 08:55:18 2020

@author: sandesh
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from kmodes.kprototypes import KPrototypes
import seaborn as sns
import os



# Running K-Prototype clustering
def model(mark_array,df):    
    kproto = KPrototypes(n_clusters=3, verbose=2, max_iter=20)
    clusters = kproto.fit_predict(mark_array, categorical=[0, 1, 2, 3])
      
    cluster_dict=[]
    for c in clusters:
        cluster_dict.append(c)
        
    df['cluster']=cluster_dict
    df_with_clusters=df
    return df_with_clusters



def main():
    st.title("EI Maven")
    html_temp = """
    <div style="background-color:#79BAEC;padding:10px">
    <h2 style="color:white;text-align:center;">Employee Assessment Engine </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
        
        
    def file_selector(folder_path='.'):
        filenames = os.listdir(folder_path)
        selected_filename = st.selectbox("Select A file",filenames)
        return os.path.join(folder_path,selected_filename)
    
               
    filename = file_selector()
    st.info("You Selected {}".format(filename))
    
    #Read data
    data = pd.read_excel(filename)
    df = data.iloc[:, [3,6,10,11,13]]
    df = df.dropna(axis=0, subset=['Task Type','Priority','Status','Assign To','Actual Person Hr'])
        
    #Show Dataset
    if st.checkbox("Show Dataset"):
        number = st.number_input("Number of Rows to View",1,10000000000)
        st.dataframe(df.head(number))
        
    # Show Columns
    if st.sidebar.button("Column Names"):
        st.write(df.columns)
        
    # Show Shape
    if st.sidebar.checkbox("Shape of Dataset"):
        data_dim = st.sidebar.radio("Show Dimension By ",("Rows","Columns"))    
        if data_dim == 'Rows':
            st.text("Number of Rows")
            st.write(df.shape[0])
        elif data_dim == 'Columns':
            st.text("Number of Columns")
            st.write(df.shape[1])
        else:
            st.write(df.shape)  
            
    # Select Columns
    if st.checkbox("Select Columns To Show"):
        all_columns = df.columns.tolist()
        selected_columns = st.multiselect("Select",all_columns)
        new_df = df[selected_columns]
        #st.dataframe(new_df)
        st.table(new_df)
        
        
    #Input Employee's Name
    employee_name = st.sidebar.text_input("Enter Employee's Name':","Type Here")
        
    st.subheader('Choose optimal K (Number of Clusters)')
    mark_array=df.values
    mark_array[:, 4] = mark_array[:, 4].astype(float)
    if st.button('Display the Elbow Method graph'):
        st.success("Generating Plot")
        cost = []
        for num_clusters in list(range(1,14)):
            kproto = KPrototypes(n_clusters=num_clusters, init='Cao')
            kproto.fit_predict(mark_array, categorical=[0,1,2,3])
            cost.append(kproto.cost_)
    
        plt.plot(cost)
        plt.xlabel('Number of Iterations')
        plt.ylabel('Cost')
        st.pyplot()
       
    
        
    st.subheader('Run the Model')
    if st.button('Run'):    
        st.success('Running the Model')
        clusters=model(mark_array,df)
        
        cluster0=clusters[clusters['cluster']== 0]
        cluster1=clusters[clusters['cluster']== 1]
        cluster2=clusters[clusters['cluster']== 2]
    
        st.write('Cluster 0 contents')
        st.table(cluster0)
        st.write('Cluster 1 contents')
        st.table(cluster1)
        st.write('Cluster 2 contents')
        st.table(cluster2)
    
        # Checking the clusters created
        value_counts = pd.DataFrame(clusters['cluster'].value_counts())
        st.write('Value Counts')
        st.write(value_counts)
        
        #Plot Bar Graph
        st.write('Overall Bar Graph Representation')
        st.write(sns.barplot(x=value_counts.index, y=value_counts['cluster']))
        st.pyplot()
        
        #Filter Employee
        # retrieving rows by individual's name by loc method 
        individual_rows=clusters.loc[clusters['Assign To'] == employee_name]
        st.write(employee_name+'\'s Evaluation')
        st.table(individual_rows)
         
        # Checking the clusters created for an individual
        value_counts_individual = pd.DataFrame(individual_rows['cluster'].value_counts())
        st.write('Value Counts for '+ employee_name)
        st.write(value_counts_individual)
        
        st.write('Bar Graph Representation of '+ employee_name+'\'s Performance')
        st.write(sns.barplot(x=value_counts_individual.index, y=value_counts_individual['cluster']))
        st.pyplot()
        
        




if __name__=='__main__':
    main()
    
