#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv('reach analysis.csv', encoding='latin1')


# In[3]:


df.head()


# In[4]:


import numpy as np

# Function to add noise to numerical columns
def augment_numerical_data(row, noise_level=1):
    augmented_row = row.copy()
    for col in numerical_columns:
        noise = np.random.normal(loc=0, scale=noise_level * row[col])
        augmented_row[col] = max(0, row[col] + noise)  # Ensure values stay non-negative
    return augmented_row

# Identify numerical and textual columns
numerical_columns = df.select_dtypes(include=np.number).columns
textual_columns = ['Caption', 'Hashtags']

# Generate synthetic data by augmenting existing rows
augmented_data = []
target_size = 1200
while len(augmented_data) + len(df) < target_size:
    random_row = df.sample(1).iloc[0]
    augmented_row = augment_numerical_data(random_row)
    
    # Augment textual data (simple augmentation by shuffling hashtags)
    augmented_row['Caption'] = random_row['Caption']  # Keeping caption as is for simplicity
    augmented_row['Hashtags'] = ' '.join(np.random.permutation(random_row['Hashtags'].split()))
    
    augmented_data.append(augmented_row)

# Convert augmented data into a DataFrame
augmented_df = pd.DataFrame(augmented_data)

# Combine original and augmented data
expanded_dataset = pd.concat([df, augmented_df], ignore_index=True)

# Check the new size of the dataset
expanded_dataset.info(), expanded_dataset.head()


# In[5]:


expanded_dataset.shape()


# In[6]:


num_rows = expanded_dataset.shape[0]
print(f"Number of rows: {num_rows}")


# In[7]:


expanded_dataset.info()


# In[8]:


expanded_dataset.head(300)


# In[ ]:


numerical_columns = expanded_dataset.select_dtypes(include=['float64', 'int64']).columns

# Convert all numerical columns to integers
expanded_dataset[numerical_columns] = expanded_dataset[numerical_columns].astype(int)

# Verify the data types
print(expanded_dataset.dtypes)


# In[ ]:


expanded_dataset.head(600)


# In[ ]:


plt.figure(figsize=(10, 8))
plt.style.use('fivethirtyeight')
plt.title("Distribution of Impressions From Home")
#sns.distplot(data['From Home'])
sns.histplot(expanded_dataset['From Home'])
plt.show()


# In[9]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
#from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveRegressor


# In[10]:


plt.figure(figsize=(10, 8))
plt.style.use('fivethirtyeight')
plt.title("Distribution of Impressions From Home")
#sns.distplot(data['From Home'])
sns.histplot(expanded_dataset['From Home'])
plt.show()


# In[11]:


plt.figure(figsize=(10, 8))
plt.title("Distribution of Impressions From Hashtags")
#sns.distplot(data['From Hashtags'])
sns.histplot(expanded_dataset['From Hashtags'])
plt.show()


# In[12]:


data = expanded_dataset


# In[13]:


home = data["From Home"].sum()
hashtags = data["From Hashtags"].sum()
explore = data["From Explore"].sum()
other = data["From Other"].sum()

labels = ['From Home','From Hashtags','From Explore','Other']
values = [home, hashtags, explore, other]

fig = px.pie(data, values=values, names=labels, 
             title='Impressions on Instagram Posts From Various Sources', hole=0.5)
fig.show()


# In[14]:


get_ipython().system('pip install wordcloud')


# In[15]:


from wordcloud import WordCloud,STOPWORDS,ImageColorGenerator


# In[16]:


text = " ".join(i for i in data.Caption)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
plt.style.use('classic')
plt.figure( figsize=(12,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[17]:


text = " ".join(i for i in data.Hashtags)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
plt.figure( figsize=(12,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[18]:


figure = px.scatter(data_frame = data, x="Impressions",
                    y="Likes", size="Likes", trendline="ols", 
                    title = "Relationship Between Likes and Impressions")
figure.show()


# In[19]:


figure = px.scatter(data_frame = data, x="Impressions",
                    y="Comments", size="Comments", trendline="ols", 
                    title = "Relationship Between Comments and Total Impressions")
figure.show()


# In[20]:


figure = px.scatter(data_frame = data, x="Impressions",
                    y="Shares", size="Shares", trendline="ols", 
                    title = "Relationship Between Shares and Total Impressions")
figure.show()


# In[21]:


correlation = data.corr()
print(correlation["Impressions"].sort_values(ascending=False))


# In[22]:


conversion_rate = (data["Follows"].sum() / data["Profile Visits"].sum()) * 100
print(conversion_rate)


# In[23]:


figure = px.scatter(data_frame = data, x="Profile Visits",
                    y="Follows", size="Follows", trendline="ols", 
                    title = "Relationship Between Profile Visits and Followers Gained")
figure.show()


# In[24]:


x = np.array(df[['Likes', 'Saves', 'Comments', 'Shares', 
                   'Profile Visits', 'Follows']])
y = np.array(df["Impressions"])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                test_size=0.2, 
                                                random_state=42)


# In[25]:


model = PassiveAggressiveRegressor()
model.fit(xtrain, ytrain)
model.score(xtest, ytest)


# In[26]:


from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error

# Load your data into the 'data' DataFrame

x = data[['Likes', 'Saves', 'Comments', 'Shares', 'Profile Visits', 'Follows']].values
y = data['Impressions'].values

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

# Hyperparameter Tuning using GridSearchCV
param_grid = {
    'C': [0.01, 0.1, 1, 10],  # Regularization parameter
    'fit_intercept': [True, False],
    'max_iter': [100, 200, 300]
}

grid_search = GridSearchCV(PassiveAggressiveRegressor(random_state=42), param_grid, cv=5)
grid_search.fit(xtrain, ytrain)

best_model = grid_search.best_estimator_

# Cross-validation
cv_scores = cross_val_score(best_model, xtrain, ytrain, cv=5)
mean_cv_score = np.mean(cv_scores)

# Model evaluation
ypred = best_model.predict(xtest)
mse = mean_squared_error(ytest, ypred)
r2_score = best_model.score(xtest, ytest)

print("Best Model:", best_model)
print("Cross-Validation Scores:", cv_scores)
print("Mean Cross-Validation Score:", mean_cv_score)
print("Mean Squared Error:", mse)
print("R-squared Score:", r2_score)


# In[27]:


# Features = [['Likes','Saves', 'Comments', 'Shares', 'Profile Visits', 'Follows']]
features = np.array([[282.0, 233.0, 4.0, 9.0, 165.0, 54.0]])
model.predict(features)


# In[28]:


df['Impressions'].value_counts()


# In[29]:


df['Impressions'].describe()


# In[30]:


# Example thresholds for classification
def classify_reach(	Impressions):
    if Impressions < 3000:
        return 'Low'
    elif Impressions < 5000:
        return 'Medium'
    else:
        return 'High'

# Apply classification
df['Impression_Class'] = df['Impressions'].apply(classify_reach)


# In[31]:


from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
df['Impression_Class_Encoded'] = encoder.fit_transform(df['Impression_Class'])
print(encoder.classes_)


# In[32]:


from sklearn.model_selection import train_test_split

# Features (drop non-numerical and target columns)
X = df[['Likes', 'Saves', 'Comments', 'Shares', 'Profile Visits', 'Follows']].values
y = df['Impression_Class_Encoded']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[33]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[34]:


from sklearn.svm import SVC

# Initialize and train the SVM model
model = SVC(kernel='linear', random_state=42)
model.fit(X_train, y_train)


# In[35]:


from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Make predictions
y_pred = model.predict(X_test)

# Print evaluation metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# In[36]:


model.score(X_test, y_test)


# In[ ]:




