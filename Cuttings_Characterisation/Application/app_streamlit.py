import pandas as pd
import time
from PIL import Image
import plotly.graph_objects as go
import random
import numpy as np
import streamlit as st

N = 20

st.set_page_config(layout="wide")

def reorder(arr, index):
    return [arr[i] for i in index] 

# Function to load CSV file and return list of image paths
def load_data(test_name, N=N):
    df = pd.read_csv(test_name, index_col=0)

    return df['Paths_Test'].tolist()[:N], df['Label'].tolist()[:N]

# Create a Streamlit SessionState object

session_state = st.session_state

if 'counter' not in session_state:
    session_state.counter = 0

if 'images' not in session_state:
    session_state.images = []

if 'labels' not in session_state:
    session_state.labels = []

if 'preds' not in session_state:
    session_state.preds = []

if 'random_idx' not in session_state:
    session_state.random_idx = random.sample(range(0,N), N)

if 'start' not in session_state:
    st.session_state.disabled = False

# Create dictionary to map test names to CSV file paths
test_csv_paths = {
    'Lab/Lab': 'data/lab/train_test_test_mar.csv',
    'Lab/Borehole': 'data/borehole/train_test_test_mar.csv',
    'Borehole/Borehole': 'data/borehole/train_test_test_mar.csv',
    'Borehole/Lab': 'data/lab/train_test_test_mar.csv'
}

# examples_csv_paths = {
#     'Lab/Lab': 'data/lab/train_test_train_mar.csv',
#     'Lab/Borehole': 'data/lab/train_test_train_mar.csv',
#     'Borehole/Borehole': 'data/borehole/train_test_train_mar.csv',
#     'Borehole/Lab': 'data/borehole/train_test_train_mar.csv'
#     }

###
col1, col2 = st.columns(2)

with col1:

    ### Explanation ###
    st.write('Explanation about how to proceed')

    ### Test selection ###
    test_name = st.selectbox('Select test', options=list(test_csv_paths.keys()))

    ### Settings##
    send_results = st.button('Send Results')
    restart = st.button('Restart')
    start = st.button('Start', key='start', disabled=st.session_state.disabled)
    print(start,'start')
    if send_results: # activated when N is reached
        pass

    if start:
        st.session_state.disabled = True

    if restart: # pop from dict

        print(session_state.keys())
        session_state.counter = 0
        session_state.preds = []
        session_state.images = []
        session_state.labels = []
    


session_state.images, session_state.labels = load_data(test_csv_paths[test_name])

print('labels',session_state.labels)
print('random_idx',session_state.random_idx)

# Sure we load the correct image 
print(session_state.counter)
print(session_state.random_idx[session_state.counter])

image = Image.open(session_state.images[session_state.random_idx[session_state.counter]])

image = image.resize((600, 600))

st.image(image, caption=f"image {session_state.counter + 1}/{len(session_state.images)}", width=400)
    
# create buttons for prediction
col1, col2, col3, col4, col5 = st.columns(5)
for i, col in enumerate([col1, col2, col3, col4, col5]):
    col.button(f'Button {i+1}',key=f'but{i+1}')

if sum([session_state.get(f'but{i+1}') for i in range(5)]):

    if session_state.get('but1'):
        session_state.preds.append((0,session_state.labels[session_state.counter]))
        session_state.counter += 1
    if session_state.get('but2'):
        session_state.preds.append((1,session_state.labels[session_state.counter]))
        session_state.counter += 1
    if session_state.get('but3'):
        session_state.preds.append((2,session_state.labels[session_state.counter]))
        session_state.counter += 1
    if session_state.get('but4'):
        session_state.preds.append((3,session_state.labels[session_state.counter]))
        session_state.counter += 1
    if session_state.get('but5'):
        session_state.preds.append((4,session_state.labels[session_state.counter]))
        session_state.counter += 1


    print(session_state.preds[-1])

    print(session_state.preds)

    print(session_state.counter)
# print(session_state.preds)
# print(f'Image {session_state.counter}:')
