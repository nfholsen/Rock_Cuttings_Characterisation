import streamlit as st
import pandas as pd
import numpy as np
import streamlit as st

session_state = st.session_state

# Create a placeholder dataframe
data = {'image': ['image1.jpg', 'image2.jpg', 'image3.jpg', 'image4.jpg'],
        'label': [0, 1, 2, 3]}
df = pd.DataFrame(data)

if 'predictions' not in session_state:
    session_state.predictions=[]

if 'counter' not in session_state:
    session_state.counter=0

# Streamlit app
def main():
    st.title("Image Classification App")

    # Choose dataframe option
    option = st.selectbox("Choose a dataframe:", ["DataFrame 1", "DataFrame 2", "DataFrame 3", "DataFrame 4"])

    if option == "DataFrame 1":
        df = pd.DataFrame({'image': ['image1.jpg', 'image2.jpg', 'image3.jpg', 'image4.jpg'],
                           'label': [0, 1, 2, 3]})
    # Define other dataframes for options 2, 3, and 4
    
    # Display dataframe
    st.write("Loaded DataFrame:")
    st.write(df)

    # Create classification buttons
    if len(df) > 0:
        image = df.iloc[0]['image']
        label = df.iloc[0]['label']
        button_label = f"Class {label}"
        if st.button(button_label):
            prediction = label
            session_state.predictions.append((image, label, prediction))
            session_state.counter += 1

            print(session_state.counter)
            df = df.iloc[1:]  # Remove the processed row
            print(df)
            st.write(f"Image: {image}, True Label: {label}, Predicted Label: {prediction}")
    else:
        st.write("All images predicted! Click 'Save CSV' to save the results.")

    # Save CSV button
    if st.button("Save CSV"):
        if len(session_state.predictions) > 0:
            results_df = pd.DataFrame(session_state.predictions, columns=['image', 'true_label', 'predicted_label'])
            results_df.to_csv("image_classification_results.csv", index=False)

if __name__ == "__main__":
    main()