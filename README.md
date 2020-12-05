# Time-Series-Analysis-of-Superstore-Sales-Data
Perform EDA and build a forecast model using the Superstore sales data 

## **Abstract**

Time series analysis deals with time series based data to extract patterns for predictions and other characteristics of the data. It uses a model for forecasting future values in a small time frame based on previous observations. It is widely used for non-stationary data, such as economic data, weather data, stock prices, and retail sales forecasting.

This readme describes how the code for our Time Series Analysis and forecasting model works.

## **Obejective** 

Our main objective is to analyze the time series data in order to extract meaningful statistics and other characteristics of the data and then build a Time series forecasting model to predict future values based on previously observed values.


## **Methodology** 

<img src = "https://github.com/HemachandarN/Time-Series-Analysis-of-Superstore-Sales-Data/blob/main/Images/Process_outline.JPG">

## **Data Source**<br>
### **Kaggle**<br>
Link: https://www.kaggle.com/rohitsahoo/sales-forecasting<br>

## **Libraries Used**
- Pandas<br>
- Numpy<br>
- Stats<br>
- Seaborn<br>
- Matplotlib<br>
- Sklearn.model_selection<br>
- Sklearn Metrics<br>
- Sklearn.pipeline<br>
- Sklearn.preprocessing<br>
- Sklearn.model_selection
- Statsmodels.tsa.arima_model 
- Fbprophet

## Column information 

We have a sales dataset of 9800 rows and 18 columns. The feature variables are:  

Columns | Definition |
 --- | --- |
 Row ID  							                   |Integer Categorical variable that refers to the specific piece being reviewed
 Order ID 					           |Integer Categorical variable that refers to the specific piece being ordered
 Order Date 		        |Date of the order
 Ship Date  					               	   	|Shipping date
 Ship Mode							                	      |Shipping Mode 
 Customer ID  					                	|Integer Categorical variable that refers to the customer 
 Customer Name 								                     |Name of the specific customer
 Segment 					           |Segment through which the order is placed
 Country 			       |Country at which the order is made from
 City 							                  |City at which the order is made from
 State 							                  |State at which the order is made from
 Postal Code 							                  |Postal code of the place of order
 Region 							                  |Region at which the order is made from
 Product ID 							                  |Integer Categorical variable that refers to the specific product being ordered
 Category 							                  |Category to which the order belongs to
 Sub-Category 							                  |Sub-category to which the order belongs to
 Product Name 							                  |Name of the product being ordered
 Sales 							                  |Price of the product in US dollars
  
  
  
## **Data Analysis and Interpretation** 

### **Exploratory Data Analysis**

In this starting stage we analyze each of the features in the dataset. Mainly the relevant ones; `Ship Mode`, `Segment`, `Country`, `City`, `State`, `Region`, `Category` and `Sub-Category` for Univariate analysis.  Here we generally use summary statistics and visualizations to gain insights on the features and generate our hypotheses for the dataset. Then we do Bivariate analysis where we do the same thing as above with two variables to check whether they have any empirical relationship between them. 

#### **Visualizations from EDA**

Some of the visualizations from the analysis are shown below. A more detailed version can be found with the code. 

<img src = "https://github.com/HemachandarN/Time-Series-Analysis-of-Superstore-Sales-Data/blob/main/Images/1.JPG">

<img src = "https://github.com/HemachandarN/Time-Series-Analysis-of-Superstore-Sales-Data/blob/main/Images/2.JPG">

<img src = "https://github.com/HemachandarN/Time-Series-Analysis-of-Superstore-Sales-Data/blob/main/Images/3.JPG">

<img src = "https://github.com/HemachandarN/Time-Series-Analysis-of-Superstore-Sales-Data/blob/main/Images/4.JPG">

<img src = "https://github.com/HemachandarN/Time-Series-Analysis-of-Superstore-Sales-Data/blob/main/Images/5.JPG">


### **Time Series Analysis** 

This mainly involves the analysis of `Sales` across the `Order Date`. We first plot the sales across the different perdiod available in order to understand the trend and seasonality. Then we split the date into month and year and calculated the mean `Sales` for each month across the different years. Below you can a see the `Sales` trend across different months for each year present in the data. 

<img src = "https://github.com/HemachandarN/Time-Series-Analysis-of-Superstore-Sales-Data/blob/main/Images/6.JPG">

### **Data Preprocessing**

- Remove missing values as they have no meaning.
- Convert the time based columns which are as `Objects` into `datetime` formats.
- The trend of the sales data should be stationary for time series models. In order to do that we can use first log difference of the sales data for the time series model to get better results. 

### **Forecastinng Models**

For forecasting `Sales` we chose two approaches
- Regression Models
- Time Serie Models 

#### **Regression Models**

Regression model is a set of statistical processes for estimating the relationships between a dependent variable (`Sales` in our case) and one or more independent variables (`Segment`, `Country`, `State` etc...).

We have used two types of regression to predict `Sales`. 
- Linear Regression  
- Polynomical Regression

#### **Time Series Models**




## **Who might be interested in this data**

Forecasting the sales could offer tremendoud insights of how the store is doing business. Especially for a superstore an upward trend in the forecast will indicate the store is doing good business. Online retails , superstores and other ecommerce platformas might be interested in this analysis. It will help the business to move forward and serve and as a basis of performance mertric. 

## **Challenges Faced**

1. The main challenge was the text preprocessing to get clean text data out. 
2. Feature selection; Feature selection here was mainly done by correlation. Doing feature selection using supervised learning algorithms would have got us better model metrics. 
3. Repetition of products; The 23,000 customer reviews were mainly focussed towards 1206 products. 
4. Title column; We tried to use it for our analysis, but we were not able to use it in the end. 

## **Limitations**

The analysis mainly focuses on the reviews and ratings given by a customer and how it plays a role in their recommendation for that product. But there can be other reasons involved in the recommendation like sale details of the product like price, delivery date, quality( they would have got something different from what they saw online). The lack of information on the impact of brand loyalty and brand likeliness affected by competition from other retailers as we do not have data about them as well.

The analysis could be improved if the products are segregated by sellers and particular products. Because this is real commercial data, it has been anonymized, and references to the company in the review text and body have been replaced with “retailer”. Including the price column will also be very helpful as mentioned. 


