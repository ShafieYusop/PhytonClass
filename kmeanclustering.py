file = "mall_customer.csv"

import streamlit as st
import pandas as pd
import numpy as np
df = pd.read_csv(file)

kmeans = KMeans(n_clusters=5)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)
st.scatter(X['Annual_Income_(k$)'], X['Spending_Score'], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
st.scatter(centers[:,0], centers[:,1], c='black', s=200, alpha=0.5)

features = ['Annual_Income_(k$)', 'Spending_Score']
X = df[features]

st.scatter(X['Annual_Income_(k$)'], X['Spending_Score']);
step_size = 0.01

x_min, x_max = min(X.iloc[:,0]) - 1, max(X.iloc[:,0]) + 1
y_min, y_max = min(X.iloc[:,1]) - 1, max(X.iloc[:,1]) + 1
x_values, y_values = np.meshgrid(np.arange(x_min,x_max,step_size), np.arange(y_min,y_max,step_size))
predictions = kmeans.predict(np.c_[x_values.ravel(), y_values.ravel()])

predictions = predictions.reshape(x_values.shape)
st.figure(figsize=(8,6))
st.imshow(predictions, interpolation='nearest', extent=(x_values.min(), x_values.max(), y_values.min(), y_values.max()), 
           cmap=st.cm.Spectral, aspect='auto', origin='lower')

st.scatter(X.iloc[:,0],X.iloc[:,1], marker='o', facecolors='grey',edgecolors='w',s=30)
centroids = kmeans.cluster_centers_
st.scatter(centroids[:,0], centroids[:,1], marker='o', s=200, linewidths=3, 
           color='k', zorder=10, facecolors='black')

st.title('Centroids and boundaries calculated using KMeans Clustering', fontsize=16)
st.show()