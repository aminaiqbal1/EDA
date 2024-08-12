import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.graph_objects as go

# Add title and subheader
st.title('Data Analysis Application')
st.subheader('For Simple Data')

# Create a dropdown list to choose a dataset
dataset_options = ['iris', 'titanic', 'tips', 'diamonds']
selected_dataset = st.selectbox('Select a dataset', dataset_options)

# Load the selected dataset
if selected_dataset == 'iris':
    df = sns.load_dataset('iris')
elif selected_dataset == 'titanic':
    df = sns.load_dataset('titanic')
elif selected_dataset == 'tips':
    df = sns.load_dataset('tips')
elif selected_dataset == 'diamonds':
    df = sns.load_dataset('diamonds')

# Button to upload custom dataset
uploaded_file = st.sidebar.file_uploader('Upload dataset', type=['csv', 'xlsx'])

if uploaded_file is not None:
    # Process the uploaded file
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error loading file: {e}")

# Display the dataset
st.write(df)

# Show data overview in the sidebar
st.sidebar.subheader("Data Overview")
st.sidebar.write(f"Number of rows: {df.shape[0]}")
st.sidebar.write(f"Number of columns: {df.shape[1]}")

# Display the column names of selected data with their data types
st.write('Column Names and Data Types:')
st.write(df.dtypes)

# Print the null values if those are > 0
if df.isnull().sum().sum() > 0:
    st.write('Null Values:', df.isnull().sum().sort_values(ascending=False))
else:
    st.write('No Null Values')

# Display the summary statistics of the selected data
st.write('Summary Statistics:')
st.write(df.describe())

# Create a pairplot
st.subheader('Pairplot')
# Select the column to be used as hue in pairplot
hue_column = st.selectbox('Select a column to be used as hue', df.columns)
st.pyplot(sns.pairplot(df, hue=hue_column))

# Create a heatmap
st.subheader('Heatmap')
# Select the columns which are numeric and then create a corr_matrix
numeric_columns = df.select_dtypes(include=np.number).columns
corr_matrix = df[numeric_columns].corr()

# Convert the seaborn heatmap plot to a Plotly figure
heatmap_fig = go.Figure(data=go.Heatmap(z=corr_matrix.values,
                                        x=corr_matrix.columns,
                                        y=corr_matrix.columns,
                                        colorscale='Viridis'))
st.plotly_chart(heatmap_fig)

# Sidebar for exporting data
st.sidebar.header('Export Data')
export_as_csv = st.sidebar.button('Export as CSV')
if export_as_csv:
    csv = df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label='Download CSV',
        data=csv,
        file_name='exported_data.csv',
        mime='text/csv',
    )