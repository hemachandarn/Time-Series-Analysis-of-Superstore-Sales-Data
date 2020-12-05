# Time-Series-Analysis-of-Superstore-Sales-Data
Perform EDA and build a forecast model using the Superstore sales data 

## **Abstract**

Customer feedback is a crucial source of information describing user experience with a company and its service. Just like in product development, efficient use of feedback can help identify and prioritize opportunities for company’s further development.

This readme describes how the code for our Text Analysis and Classification Model works.

## **Obejective** 

In our analysis, we will utilize the power of text mining to do an in-depth analysis of customer reviews on an e-commerce clothing site data and build a classification model to predict whether the customer will recommend the product or not.

It will help retailers to have an understanding about their products, mistakes and customer satisfaction.


## **Methodology** 

<img src = "https://github.com/HemachandarN/Women-s-E-Commerce-Clothing-Reviews/blob/master/Data/Process_Outline.PNG">

## **Data Source**<br>
### **Kaggle**<br>
Link: https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews<br>

## **Libraries Used**
import pandas as pd<br>
import numpy as np<br>
from scipy import stats<br>
import string<br>
import seaborn as sns<br>
import matplotlib.pyplot as plt<br>
import matplotlib.gridspec as gridspec<br>
from sklearn.model_selection import train_test_split as split<br>
from sklearn import metrics<br> 
from sklearn.metrics import classification_report, confusion_matrix, auc, roc_curve<br>
import nltk<br>
from nltk.corpus import stopwords<br>
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer <br>
from nltk.stem import PorterStemmer, LancasterStemmer<br>
from sklearn.feature_extraction.text import CountVectorizer<br>
from sklearn.feature_extraction.text import TfidfTransformer<br>
from nltk.tokenize import word_tokenize<br>
from nltk.probability import FreqDist<br>
import spacy<br>
import re<br>
from wordcloud import WordCloud<br>
from imblearn.over_sampling import SMOTE<br>
from nltk.stem.snowball import SnowballStemmer<br>
from nltk.sentiment.vader import SentimentIntensityAnalyzer<br>
from sklearn.tree import DecisionTreeClassifier<br>
from sklearn.linear_model import LogisticRegression<br>
from sklearn.naive_bayes import GaussianNB<br>
from sklearn import svm<br>
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score<br>
import warnings<br>
warnings.filterwarnings('ignore') <br>
from IPython.display import Image<br>
%matplotlib inline<br>

## Column information 

We have a review dataset of 23486 rows and 10 columns of data. The feature variables are:  

Columns | Definition |
 --- | --- |
 Clothing ID  							                   |Integer Categorical variable that refers to the specific piece being reviewed
 Age 					           |Positive Integer variable of the reviewers age
 Title 		        |String variable for the title of the review
 Review Text  					               	   	|String variable for the review body
 Rating							                	      |String variable for the review body 
 Recommended IND  					                	|Binary variable stating where the customer recommends the product where 1 is recommended, 0 is not recommended 
 Positive Feedback Count 								                     |Positive Integer documenting the number of other customers who found this review positive
 Division Name 					           |Categorical name of the product high level division
 Department Name 			       |Categorical name of the product department name
 Class Name 							                  |Categorical name of the product class name
 
## **Data Analysis and Interpretation** 

### **Exploratory Data Analysis**

In this starting stage we analyze each of the features in the dataset. Mainly the relevant ones; continuous columns Age and Positive Feedback; Binary Column Recommendations IND and Categorical Columns Rating, Department Name, Class Name and Division Name. Here we generally use summary statistics and visualizations to gain insights on the features and generate our hypotheses for the dataset. Then we do Bivariate analysis where we do the same thing as above with two variables to check whether they have any empirical relationship between them. 

### **Data Preprocessing** 

This involves a lot of steps. Analyzing the data that has not been carefully screened will lead to bad results. We need to prepare the data for the analysis and data preprocessing is the way. This process involves handling missing values, Removing outliers that are not useful and may skew the data, removing redundant features that are not useful for our analysis, encoding categorical variables, balancing the dataset and finally feature selection. 

### **Text Preprocessing**

Text Preprocessing comes under data preprocessing. This is step mainly deals with text data. Here we clean the text data so that it will be easier to do analysis on them. The steps involved in this cleaning process are removal of stopwords, removal of punctuations, converting the words to lowercase and lemmatization. After the cleaning process the review is turned into individual words/tokens from which we can find out frequent words used by the people who give good ratings and vice versa.  

### **Sentiment Analysis**

Sentiment Analysis is used to systematically identify, extract, quantify, and study affective states and subjective information. This analysis basically gives out the emotion behind the customer review in the form of a polarity score. Polarity is nothing but a float that lies between -1 to 1. This tells us whether the review is positive, negative or neutral. We classify each of the reviews using this the polarity score into these three categories. 

### **Classification Models** 

Based on the detailed analysis on the customer review text data and other numerical, binary and categorical data we build our classification models. The target variable here as mentioned above is whether the customer will recommend the product or not. We have used 4 models Decision Tree, Logistic Regression, Support Vector Machine and Naive Bayes. The metrics for the Logistics regression model were the best.  


## **Who might be interested in this data**

Customer reviews are a great source of “Voice of the customer” and could offer tremendous insights into what customers like and dislike about a product or service. For the e-commerce business, customer reviews are very critical, since existing reviews heavily influence buying decision of new customers in the absence of the actual look and feel of the product to be purchased.

Online retailers might be interested on this analysis. It will serve them as a basis towards a bigger picture of **customer retention** as well. 

## **Challenges Faced**

1. The main challenge was the text preprocessing to get clean text data out. 
2. Feature selection; Feature selection here was mainly done by correlation. Doing feature selection using supervised learning algorithms would have got us better model metrics. 
3. Repetition of products; The 23,000 customer reviews were mainly focussed towards 1206 products. 
4. Title column; We tried to use it for our analysis, but we were not able to use it in the end. 

## **Limitations**

The analysis mainly focuses on the reviews and ratings given by a customer and how it plays a role in their recommendation for that product. But there can be other reasons involved in the recommendation like sale details of the product like price, delivery date, quality( they would have got something different from what they saw online). The lack of information on the impact of brand loyalty and brand likeliness affected by competition from other retailers as we do not have data about them as well.

The analysis could be improved if the products are segregated by sellers and particular products. Because this is real commercial data, it has been anonymized, and references to the company in the review text and body have been replaced with “retailer”. Including the price column will also be very helpful as mentioned. 


