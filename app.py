import streamlit as st
from pathlib import Path
import base64
import requests

# Initial page config
st.set_page_config(
    page_title='📊 Comprehensive Data Science Cheat Sheet',
    layout="wide",
    initial_sidebar_state="expanded",
)

def main():
    ds_sidebar()
    ds_body()

# Function to convert image to base64 bytes (for logo)
def img_to_bytes(img_url):
    try:
        response = requests.get(img_url)
        img_bytes = response.content
        encoded = base64.b64encode(img_bytes).decode()
        return encoded
    except:
        return ''

# Sidebar content
def ds_sidebar():
    logo_url = 'https://ahammadmejbah.com/content/images/2024/10/Mejbah-Ahammad-Profile-8.png'
    logo_encoded = img_to_bytes(logo_url)
    
    st.sidebar.markdown(
        f"""
        <a href="https://ahammadmejbah.com/">
            <img src='data:image/png;base64,{logo_encoded}' class='img-fluid' width=100>
        </a>
        """,
        unsafe_allow_html=True
    )
    st.sidebar.header('🧰 Data Science Cheat Sheet')

    st.sidebar.markdown('''
    <small>Comprehensive summary of essential Data Science concepts, libraries, and tools.</small>
    ''', unsafe_allow_html=True)

    st.sidebar.markdown('__🔑 Key Libraries__')
    st.sidebar.code('''
$ pip install numpy pandas matplotlib seaborn scikit-learn tensorflow pytorch nltk spacy
    ''')

    st.sidebar.markdown('__💻 Common Commands__')
    st.sidebar.code('''
$ jupyter notebook
$ python script.py
$ git clone https://github.com/ahammadmejbah/Data-Science-Cheat-Sheet
$ streamlit run app.py
    ''')

    st.sidebar.markdown('__🔄 Data Science Workflow__')
    st.sidebar.code('''
1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis
4. Feature Engineering
5. Model Building
6. Evaluation
7. Deployment
    ''')

    st.sidebar.markdown('__💡 Tips & Tricks__')
    st.sidebar.code('''
- Use virtual environments
- Version control with Git
- Document your code
- Continuous learning
- Utilize Jupyter Notebooks for exploration
    ''')

    st.sidebar.markdown('''<hr>''', unsafe_allow_html=True)
    st.sidebar.markdown('''<small>[Cheat sheet v1.0](https://github.com/ahammadmejbah/Data-Science-Cheat-Sheet) | Nov 2024 | [Mejbah Ahammad](https://ahammadmejbah.com/)</small>''', unsafe_allow_html=True)

# Main body of cheat sheet
def ds_body():
    # Custom CSS for styling
    st.markdown("""
        <style>
            /* Header Styling */
            .header {
                background: linear-gradient(90deg, #FF4B4B, #FF9068);
                padding: 20px;
                text-align: center;
                border-radius: 10px;
                margin-bottom: 20px;
            }
            .header h1 {
                color: #FFFFFF;
                font-size: 2.5em;
                margin: 0;
                font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            }
            /* Section Header Styling */
            .section-header {
                color: #FF4B4B;
                font-size: 1.5em;
                margin-top: 20px;
                margin-bottom: 10px;
            }
            /* Code Block Styling */
            pre {
                background-color: #2E3440 !important;
                color: #D8DEE9 !important;
                padding: 10px;
                border-radius: 5px;
                font-size: 0.9em;
                overflow-x: auto;
            }
            /* Footer Styling */
            .footer {
                background-color: #2E3440;
                color: white;
                text-align: center;
                padding: 20px;
                margin-top: 50px;
                border-top: 2px solid #FF4B4B;
            }
            .social-icons img {
                width: 30px;
                margin: 0 10px;
                transition: transform 0.2s;
            }
            .social-icons img:hover {
                transform: scale(1.1);
            }
            /* Responsive Design */
            @media (max-width: 1200px) {
                .section-header {
                    font-size: 1.3em;
                }
                pre {
                    font-size: 0.85em;
                }
            }
            @media (max-width: 768px) {
                .header h1 {
                    font-size: 2em;
                }
                .section-header {
                    font-size: 1.2em;
                }
                pre {
                    font-size: 0.8em;
                }
                .social-icons img {
                    width: 25px;
                    margin: 0 5px;
                }
            }
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown(f"""
        <div class="header">
            <h1>📊 Comprehensive Data Science Cheat Sheet</h1>
        </div>
    """, unsafe_allow_html=True)

    # Create three columns
    col1, col2, col3 = st.columns(3)

    # Distribute sections among the three columns
    with col1:
        # Section 1: Python Basics
        st.markdown('<div class="section-header">🔍 Python Basics</div>', unsafe_allow_html=True)
        st.markdown('**Importing Libraries**')
        st.code('''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
        ''', language='python')

        st.markdown('**Data Structures**')
        st.code('''
# List
my_list = [1, 2, 3, 4]

# Tuple
my_tuple = (1, 2, 3, 4)

# Dictionary
my_dict = {'key1': 'value1', 'key2': 'value2'}

# Set
my_set = {1, 2, 3, 4}
        ''', language='python')

        st.markdown('**Control Flow**')
        st.code('''
# If-Else
if condition:
    # do something
elif another_condition:
    # do something else
else:
    # default action

# For Loop
for i in range(10):
    print(i)

# While Loop
while condition:
    # do something
    break
        ''', language='python')

        st.markdown('**Functions**')
        st.code('''
def my_function(param1, param2):
    """
    Function description.
    """
    result = param1 + param2
    return result
        ''', language='python')

        st.markdown('**List Comprehensions**')
        st.code('''
squares = [x**2 for x in range(10)]
even_squares = [x**2 for x in range(10) if x % 2 == 0]
        ''', language='python')

    with col2:
        # Section 2: Data Manipulation
        st.markdown('<div class="section-header">📁 Data Manipulation</div>', unsafe_allow_html=True)
        st.markdown('**Pandas Basics**')
        st.code('''
import pandas as pd

# Create DataFrame
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'Los Angeles', 'Chicago']
})

# Read CSV
df = pd.read_csv('data.csv')

# View DataFrame
df.head()
        ''', language='python')

        st.markdown('**Data Selection**')
        st.code('''
# Select column
df['Age']

# Select multiple columns
df[['Name', 'Age']]

# Select rows by index
df.iloc[0:5]

# Select rows by condition
df[df['Age'] > 30]
        ''', language='python')

        st.markdown('**Data Cleaning**')
        st.code('''
# Handle missing values
df.dropna(inplace=True)
df.fillna(value=0, inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Data type conversion
df['Age'] = df['Age'].astype(int)

# Rename columns
df.rename(columns={'Name': 'Full Name'}, inplace=True)
        ''', language='python')

        st.markdown('**Data Transformation**')
        st.code('''
# Apply function
df['Age'] = df['Age'].apply(lambda x: x + 1)

# Vectorized operations
df['Age'] = df['Age'] + 1

# Mapping
df['City'] = df['City'].map({'New York': 'NY', 'Los Angeles': 'LA', 'Chicago': 'CHI'})
        ''', language='python')

        st.markdown('**Merging & Joining**')
        st.code('''
# Merge DataFrames
merged_df = pd.merge(df1, df2, on='Key')

# Concatenate DataFrames
concatenated_df = pd.concat([df1, df2], axis=0)

# Join DataFrames
joined_df = df1.join(df2, how='inner')
        ''', language='python')

    with col3:
        # Section 3: Data Visualization
        st.markdown('<div class="section-header">📈 Data Visualization</div>', unsafe_allow_html=True)
        st.markdown('**Matplotlib**')
        st.code('''
import matplotlib.pyplot as plt

# Line Plot
plt.plot(x, y)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Line Plot')
plt.show()

# Bar Chart
plt.bar(categories, values)
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Bar Chart')
plt.show()

# Scatter Plot
plt.scatter(x, y)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scatter Plot')
plt.show()

# Histogram
plt.hist(data, bins=10)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.show()
        ''', language='python')

        st.markdown('**Seaborn**')
        st.code('''
import seaborn as sns

# Scatter Plot with Regression Line
sns.lmplot(x='Age', y='Salary', data=df)

# Heatmap
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')

# Boxplot
sns.boxplot(x='City', y='Age', data=df)

# Pairplot
sns.pairplot(df)
        ''', language='python')

        st.markdown('**Plotly**')
        st.code('''
import plotly.express as px

# Scatter Plot
fig = px.scatter(df, x='Age', y='Salary', color='City')
fig.show()

# Bar Chart
fig = px.bar(df, x='City', y='Sales', barmode='group')
fig.show()

# Line Chart
fig = px.line(df, x='Date', y='Sales', title='Sales Over Time')
fig.show()

# Histogram
fig = px.histogram(df, x='Age', nbins=10)
fig.show()
        ''', language='python')

    # Second row of columns
    col4, col5, col6 = st.columns(3)

    with col4:
        # Section 4: Machine Learning
        st.markdown('<div class="section-header">🤖 Machine Learning</div>', unsafe_allow_html=True)
        st.markdown('**Scikit-learn Basics**')
        st.code('''
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Split data
X = df[['Age', 'Experience']]
y = df['Salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f'MSE: {mse}, R2: {r2}')
        ''', language='python')

        st.markdown('**Classification Example**')
        st.code('''
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Initialize and train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(report)
        ''', language='python')

        st.markdown('**Cross-Validation**')
        st.code('''
from sklearn.model_selection import cross_val_score

# 5-Fold Cross-Validation
scores = cross_val_score(model, X, y, cv=5)
print(f'Cross-Validation Scores: {scores}')
print(f'Average CV Score: {scores.mean()}')
        ''', language='python')

        st.markdown('**Hyperparameter Tuning with GridSearchCV**')
        st.code('''
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring='accuracy')

# Fit GridSearch
grid_search.fit(X_train, y_train)

# Best parameters
print(grid_search.best_params_)

# Best score
print(grid_search.best_score_)
        ''', language='python')

    with col5:
        st.markdown('**Feature Scaling**')
        st.code('''
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
        ''', language='python')

        st.markdown('**Handling Categorical Variables**')
        st.code('''
# One-Hot Encoding
X = pd.get_dummies(X, columns=['Category'])

# Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X['Category'] = le.fit_transform(X['Category'])
        ''', language='python')

        # Additional content can be added here as needed.

    with col6:
        st.markdown('<div class="section-header">🧠 Deep Learning</div>', unsafe_allow_html=True)
        st.markdown('**TensorFlow/Keras Basics**')
        st.code('''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define the model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(input_dim,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')
        ''', language='python')

        st.markdown('**PyTorch Basics**')
        st.code('''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define the model
class MyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = MyModel(input_dim=10, hidden_dim=64, output_dim=1)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# DataLoader
dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
for epoch in range(100):
    for data, targets in loader:
        outputs = model(data)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
        ''', language='python')

        st.markdown('**Convolutional Neural Networks (CNN)**')
        st.code('''
from tensorflow.keras import layers, models

# Define CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, channels)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Compile and train
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
        ''', language='python')

    # Third row of columns
    col7, col8, col9 = st.columns(3)

    with col7:
        # Section 5: Statistical Analysis
        st.markdown('<div class="section-header">📊 Statistical Analysis</div>', unsafe_allow_html=True)
        st.markdown('**Descriptive Statistics**')
        st.code('''
# Summary statistics
df.describe()

# Mean, Median, Mode
df['Age'].mean()
df['Age'].median()
df['Age'].mode()
        ''', language='python')

        st.markdown('**Probability Distributions**')
        st.code('''
import numpy as np
import matplotlib.pyplot as plt

# Normal Distribution
data = np.random.normal(loc=0, scale=1, size=1000)
plt.hist(data, bins=30)
plt.show()

# Binomial Distribution
data = np.random.binomial(n=10, p=0.5, size=1000)
plt.hist(data, bins=30)
plt.show()
        ''', language='python')

        st.markdown('**Hypothesis Testing**')
        st.code('''
from scipy import stats

# T-Test
t_stat, p_val = stats.ttest_ind(group1, group2)

# Chi-Square Test
chi2, p, dof, ex = stats.chi2_contingency(table)

# ANOVA
f_stat, p_val = stats.f_oneway(group1, group2, group3)
        ''', language='python')

    with col8:
        st.markdown('**Correlation Analysis**')
        st.code('''
# Pearson Correlation
pearson_corr = df['A'].corr(df['B'])

# Spearman Correlation
spearman_corr = df['A'].corr(df['B'], method='spearman')

# Kendall Correlation
kendall_corr = df['A'].corr(df['B'], method='kendall')
        ''', language='python')

        st.markdown('**Confidence Intervals**')
        st.code('''
import scipy.stats as st

# 95% Confidence Interval for the mean
confidence = 0.95
n = len(data)
mean = np.mean(data)
stderr = stats.sem(data)
h = stderr * st.t.ppf((1 + confidence) / 2., n-1)
print(f'Confidence Interval: {mean-h} to {mean+h}')
        ''', language='python')

        # Additional content can be added here as needed.

    with col9:
        st.markdown('<div class="section-header">🔧 Data Engineering</div>', unsafe_allow_html=True)
        st.markdown('**SQL Basics**')
        st.code('''
-- Select statement
SELECT column1, column2 FROM table_name;

-- Where clause
SELECT * FROM table_name WHERE condition;

-- Join
SELECT a.column1, b.column2
FROM table_a a
JOIN table_b b ON a.id = b.a_id;

-- Group By
SELECT column, COUNT(*)
FROM table
GROUP BY column;

-- Order By
SELECT * FROM table ORDER BY column DESC;
        ''', language='sql')

        st.markdown('**Database Connections with SQLAlchemy**')
        st.code('''
from sqlalchemy import create_engine
import pandas as pd

# Create engine
engine = create_engine('postgresql://user:password@localhost:5432/mydatabase')

# Read SQL query into DataFrame
df = pd.read_sql('SELECT * FROM table_name', engine)

# Write DataFrame to SQL
df.to_sql('table_name', engine, if_exists='replace', index=False)
        ''', language='python')

        st.markdown('**ETL Processes**')
        st.code('''
# Extract, Transform, Load (ETL) example using Pandas
import pandas as pd

# Extract
df = pd.read_csv('data.csv')

# Transform
df['Age'] = df['Age'].fillna(df['Age'].mean())

# Load
df.to_csv('clean_data.csv', index=False)
        ''', language='python')

        st.markdown('**Data Pipelines with Airflow**')
        st.code('''
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime

def extract():
    # Extraction logic
    pass

def transform():
    # Transformation logic
    pass

def load():
    # Loading logic
    pass

default_args = {
    'start_date': datetime(2024, 1, 1),
}

with DAG('etl_pipeline', default_args=default_args, schedule_interval='@daily') as dag:
    extract_task = PythonOperator(task_id='extract', python_callable=extract)
    transform_task = PythonOperator(task_id='transform', python_callable=transform)
    load_task = PythonOperator(task_id='load', python_callable=load)

    extract_task >> transform_task >> load_task
        ''', language='python')

    # Fourth row of columns
    col10, col11, col12 = st.columns(3)

    with col10:
        # Section 6: Tools & Utilities
        st.markdown('<div class="section-header">🛠 Tools & Utilities</div>', unsafe_allow_html=True)
        st.markdown('**Virtual Environments with venv**')
        st.code('''
# Create virtual environment
python -m venv myenv

# Activate virtual environment
# On Windows
myenv\\Scripts\\activate
# On macOS/Linux
source myenv/bin/activate

# Deactivate
deactivate
        ''', language='bash')

        st.markdown('**Package Management with pip**')
        st.code('''
# Install a package
pip install package_name

# List installed packages
pip list

# Freeze requirements
pip freeze > requirements.txt

# Install from requirements
pip install -r requirements.txt
        ''', language='bash')

        st.markdown('**Docker Basics**')
        st.code('''
# Pull an image
docker pull python:3.8

# Run a container
docker run -it python:3.8 bash

# Build an image from Dockerfile
docker build -t myimage .

# List running containers
docker ps

# Stop a container
docker stop container_id
        ''', language='bash')

    with col11:
        st.markdown('**Jupyter Notebook Shortcuts**')
        st.code('''
# Create a new notebook
jupyter notebook

# Keyboard Shortcuts
- Shift + Enter: Run cell and move to next
- Ctrl + Enter: Run cell
- A: Insert cell above
- B: Insert cell below
- M: Convert to Markdown
- Y: Convert to Code
        ''', language='text')

        st.markdown('**Git Commands**')
        st.code('''
# Initialize repository
git init

# Clone repository
git clone https://github.com/ahammadmejbah/Data-Science-Cheat-Sheet.git

# Check status
git status

# Add changes
git add .

# Commit changes
git commit -m "Commit message"

# Push to remote
git push origin main

# Pull from remote
git pull origin main
        ''', language='bash')

        # Additional content can be added here as needed.

    with col12:
        # Section 7: Web Scraping
        st.markdown('<div class="section-header">🌐 Web Scraping</div>', unsafe_allow_html=True)
        st.markdown('**BeautifulSoup Basics**')
        st.code('''
import requests
from bs4 import BeautifulSoup

# Send GET request
response = requests.get('https://example.com')

# Parse HTML
soup = BeautifulSoup(response.text, 'html.parser')

# Find elements
titles = soup.find_all('h2')

for title in titles:
    print(title.get_text())
        ''', language='python')

        st.markdown('**Scrapy Framework**')
        st.code('''
import scrapy

class ExampleSpider(scrapy.Spider):
    name = 'example'
    start_urls = ['https://example.com']

    def parse(self, response):
        for title in response.css('h2::text').getall():
            yield {'title': title}

# To run the spider
# scrapy runspider example_spider.py -o output.json
        ''', language='python')

        st.markdown('**Handling JavaScript with Selenium**')
        st.code('''
from selenium import webdriver
from selenium.webdriver.common.by import By
import time

# Initialize WebDriver
driver = webdriver.Chrome(executable_path='path_to_chromedriver')

# Open URL
driver.get('https://example.com')

# Wait for JavaScript to load
time.sleep(5)

# Extract content
titles = driver.find_elements(By.TAG_NAME, 'h2')
for title in titles:
    print(title.text)

# Close browser
driver.quit()
        ''', language='python')

    # Fifth row of columns
    col13, col14, col15 = st.columns(3)

    with col13:
        # Section 8: Version Control
        st.markdown('<div class="section-header">📝 Version Control</div>', unsafe_allow_html=True)
        st.markdown('**Basic Git Commands**')
        st.code('''
# Initialize repository
git init

# Clone repository
git clone https://github.com/ahammadmejbah/Data-Science-Cheat-Sheet.git

# Check status
git status

# Add changes
git add .

# Commit changes
git commit -m "Commit message"

# Push to remote
git push origin main

# Pull from remote
git pull origin main
        ''', language='bash')

        st.markdown('**Branching**')
        st.code('''
# Create a new branch
git branch feature-branch

# Switch to the branch
git checkout feature-branch

# Create and switch
git checkout -b new-feature

# Merge branch
git checkout main
git merge feature-branch

# Delete branch
git branch -d feature-branch
        ''', language='bash')

        st.markdown('**Stashing Changes**')
        st.code('''
# Stash changes
git stash

# Apply stashed changes
git stash apply

# List stashes
git stash list

# Drop a stash
git stash drop stash@{0}
        ''', language='bash')

        st.markdown('**Resolving Conflicts**')
        st.code('''
# After a merge conflict, edit the files to resolve
# Then add and commit
git add conflicted_file.py
git commit -m "Resolved merge conflict in conflicted_file.py"
        ''', language='bash')

    with col14:
        # Section 9: Cloud Services
        st.markdown('<div class="section-header">☁️ Cloud Services</div>', unsafe_allow_html=True)
        st.markdown('**AWS Basics**')
        st.code('''
# Install AWS CLI
pip install awscli

# Configure AWS CLI
aws configure

# List S3 buckets
aws s3 ls

# Upload a file to S3
aws s3 cp local_file.txt s3://mybucket/
        ''', language='bash')

        st.markdown('**Google Cloud Platform (GCP) Basics**')
        st.code('''
# Install Google Cloud SDK
# Initialize
gcloud init

# List projects
gcloud projects list

# Deploy to App Engine
gcloud app deploy
        ''', language='bash')

        st.markdown('**Microsoft Azure Basics**')
        st.code('''
# Install Azure CLI
# Login
az login

# List resource groups
az group list

# Create a resource group
az group create --name myResourceGroup --location eastus
        ''', language='bash')

        st.markdown('**Deploying Models to AWS SageMaker**')
        st.code('''
import boto3
import sagemaker
from sagemaker import get_execution_role

# Initialize session
sagemaker_session = sagemaker.Session()
role = get_execution_role()

# Define model
from sagemaker.sklearn import SKLearnModel
model = SKLearnModel(model_data='s3://path-to-model/model.tar.gz',
                    role=role,
                    entry_point='inference.py')

# Deploy model
predictor = model.deploy(instance_type='ml.m4.xlarge', initial_instance_count=1)
        ''', language='python')

    with col15:
        # Section 10: Time Series
        st.markdown('<div class="section-header">📅 Time Series</div>', unsafe_allow_html=True)
        st.markdown('**Time Series Decomposition**')
        st.code('''
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('timeseries.csv', parse_dates=['Date'], index_col='Date')

# Decompose
decomposition = seasonal_decompose(df['Value'], model='additive')
decomposition.plot()
plt.show()
        ''', language='python')

        st.markdown('**ARIMA Modeling**')
        st.code('''
from statsmodels.tsa.arima.model import ARIMA

# Fit ARIMA model
model = ARIMA(df['Value'], order=(1,1,1))
model_fit = model.fit()

# Summary
print(model_fit.summary())

# Forecast
forecast = model_fit.forecast(steps=10)
print(forecast)
        ''', language='python')

        st.markdown('**Prophet Forecasting**')
        st.code('''
from fbprophet import Prophet

# Prepare data
df_prophet = df.reset_index().rename(columns={'Date': 'ds', 'Value': 'y'})

# Initialize and fit
model = Prophet()
model.fit(df_prophet)

# Create future dataframe
future = model.make_future_dataframe(periods=30)

# Predict
forecast = model.predict(future)

# Plot
model.plot(forecast)
plt.show()
        ''', language='python')

        st.markdown('**Rolling Statistics**')
        st.code('''
# Moving Average
df['MA'] = df['Value'].rolling(window=12).mean()

# Moving Standard Deviation
df['STD'] = df['Value'].rolling(window=12).std()
        ''', language='python')

    # Fourth row of columns
    col16, col17, col18 = st.columns(3)

    with col16:
        # Section 11: Data Pipelines
        st.markdown('<div class="section-header">🔄 Data Pipelines</div>', unsafe_allow_html=True)
        st.markdown('**Scikit-learn Pipelines**')
        st.code('''
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Define pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression())
])

# Fit pipeline
pipeline.fit(X_train, y_train)

# Predict
predictions = pipeline.predict(X_test)
        ''', language='python')

        st.markdown('**FeatureUnion for Parallel Processing**')
        st.code('''
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

# Define FeatureUnion
features = FeatureUnion([
    ('pca', PCA(n_components=2)),
    ('select', SelectKBest(k=1))
])

# Integrate into pipeline
pipeline = Pipeline([
    ('features', features),
    ('clf', LogisticRegression())
])

pipeline.fit(X_train, y_train)
        ''', language='python')

        st.markdown('**Custom Transformers**')
        st.code('''
from sklearn.base import BaseEstimator, TransformerMixin

class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, param=1):
        self.param = param
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Custom transformation logic
        return X * self.param

# Use in pipeline
pipeline = Pipeline([
    ('custom', CustomTransformer(param=2)),
    ('clf', LogisticRegression())
])
        ''', language='python')

    with col17:
        # Section 12: Deployment
        st.markdown('<div class="section-header">🚀 Deployment</div>', unsafe_allow_html=True)
        st.markdown('**Saving and Loading Models**')
        st.code('''
import joblib
import pickle

# Save with joblib
joblib.dump(model, 'model.joblib')

# Load with joblib
model = joblib.load('model.joblib')

# Save with pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load with pickle
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
        ''', language='python')

        st.markdown('**Deploying with Streamlit**')
        st.code('''
# Create a simple Streamlit app to deploy the model
import streamlit as st
import joblib

# Load model
model = joblib.load('model.joblib')

# Input features
feature1 = st.number_input('Feature 1')
feature2 = st.number_input('Feature 2')
# Add more features as needed

# Predict
if st.button('Predict'):
    prediction = model.predict([[feature1, feature2]])
    st.write(f'Prediction: {prediction[0]}')
        ''', language='python')

        st.markdown('**Deploying with Flask**')
        st.code('''
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = model.predict([data['features']])
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
        ''', language='python')

    with col18:
        st.markdown('**Deploying with Docker**')
        st.code('''
# Dockerfile
FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

CMD ["streamlit", "run", "app.py"]
        ''', language='dockerfile')

        st.markdown('**Deploying to AWS Elastic Beanstalk**')
        st.code('''
# Initialize Elastic Beanstalk
eb init -p python-3.8 my-app

# Create environment and deploy
eb create my-app-env

# Open the app
eb open
        ''', language='bash')

    # Footer with social media links
    st.markdown(f"""
        <div class="footer">
            <p>Connect with me:</p>
            <div class="social-icons">
                <a href="https://facebook.com/ahammadmejbah" target="_blank">
                    <img src="https://cdn-icons-png.flaticon.com/512/733/733547.png" alt="Facebook" />
                </a>
                <a href="https://instagram.com/ahammadmejbah" target="_blank">
                    <img src="https://cdn-icons-png.flaticon.com/512/733/733558.png" alt="Instagram" />
                </a>
                <a href="https://github.com/ahammadmejbah" target="_blank">
                    <img src="https://cdn-icons-png.flaticon.com/512/733/733553.png" alt="GitHub" />
                </a>
                <a href="https://ahammadmejbah.com/" target="_blank">
                    <img src="https://cdn-icons-png.flaticon.com/512/919/919827.png" alt="Portfolio" />
                </a>
            </div>
            <br>
            <small>Cheat sheet v1.0 | Nov 2024 | <a href="https://ahammadmejbah.com/" style="color: #FF4B4B;">Mejbah Ahammad</a></small>
        </div>
    """, unsafe_allow_html=True)

    # Optional: Add some spacing at the bottom
    st.markdown("<br><br><br>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
