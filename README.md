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

Regression model is a set of statistical processes for estimating the relationships between a dependent variable (`Sales` in our case) and one or more independent variables (`Segment`, `Country`, `State` etc.).

**Reason:** We wanted to find out how sales was affected by other factors in the dataset. Whether it was due to the region, product category or may be due to a group of customers. Forecasting Sales based using regression will give us the extra bit of information which businesses can use for further analysis.  

We have used two types of regression to predict `Sales`. 
- **Linear Regression** is based on the assumption that there is a linear relationship between both the dependent and independent variables. It also assumes that there is no major correlation between the independent variables. Multi Linear regressions can be linear and nonlinear. It has one y and two or more x variables or one dependent variable and two or more independent variables 

- **Polynomical Regression** is a one of the types of linear regression in which the relationship between the independent variable x and dependent variable y is modeled as an nth degree polynomial. Polynomial regression fits a nonlinear relationship between the value of x and the corresponding conditional mean of y, denoted E (y |x).

 **Y=θo + θ₁X + θ₂X² + … + θₘXᵐ + residual error**

#### **Time Series Models**

**Reason:** Time is the most important independent variable in any prediction of something that happens in the future. Time series models use information regarding historical values and associated patterns to predict future activity.  

##### **Autoregressive Integrated Moving Average (ARIMA)**

The Autoregressive Integrated Moving Average (ARIMA) method models the next step in the sequence as a linear function of the differenced observations and residual errors at prior time steps.It combines both Autoregression (AR) and Moving Average (MA) models as well as a differencing pre-processing step of the sequence to make the sequence stationary, called integration (I). The notation for the model involves specifying the order for the AR(p), I(d), and MA(q) models as parameters to an ARIMA function, e.g. ARIMA(p, d, q). An ARIMA model can also be used to develop AR, MA, and ARMA models.

##### **Seasonal Autoregressive Integrated Moving-Average with Exogenous Regressors (SARIMAX)**

The Seasonal Autoregressive Integrated Moving-Average with Exogenous Regressors (SARIMAX) is an extension of the SARIMA model that also includes the modeling of exogenous variables.Exogenous variables are also called covariates and can be thought of as parallel input sequences that have observations at the same time steps as the original series. The primary series may be referred to as endogenous data to contrast it from the exogenous sequence(s). The observations for exogenous variables are included in the model directly at each time step and are not modeled in the same way as the primary endogenous sequence (e.g. as an AR, MA, etc. process).The SARIMAX method can also be used to model the subsumed models with exogenous variables, such as ARX, MAX, ARMAX, and ARIMAX.

##### **Facebook Prophet**

Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. The procedure makes use of a decomposable time series model with three main model components: trend, seasonality, and holidays. Similar to a generalized additive model (GAM), with time as a regressor, Prophet fits several linear and non-linear functions of time as components. Modeling seasonality as an additive component is the same approach taken by exponential smoothing… The GAM formulation has the advantage that it decomposes easily and accommodates new components as necessary, for instance when a new source of seasonality is identified. Prophet is “framing the forecasting problem as a curve-fitting exercise” rather than looking explicitly at the time based dependence of each observation.

( A detailed version of the models tuned with their parameters is given in the jupyter notebook)

## **Model Comparison**

The following bar plot shows the comparison between our 5 models based on RMSE value. As you can see ARIMA has less RMSE and came out as the top model for the `Sales` forecast.

<img src = "https://github.com/HemachandarN/Time-Series-Analysis-of-Superstore-Sales-Data/blob/main/Images/model_comp.jpeg">

## **Who might be interested in this data**

Forecasting the sales could offer tremendoud insights of how the store is doing business. Especially for a superstore an upward trend in the forecast will indicate the store is doing good business. Online retails , superstores and other ecommerce platformas might be interested in this analysis. It will help the business to move forward and serve and as a basis of performance mertric. 

## **Challenges Faced**

1. The main challenge was deciding the time format to forecast whether use forecast daily, monthly or yearly. Yearly was out of option as only 4 years were given. Averaging the sales on a monthly basis and forecasting would have given a different reult may be even better result.  
2. Feature selection for the regression models. 
3. Parameter tuning for the time series models were difficult. For example If you take ARIMA model once you get p,d,q. You have to try all the different combinations as possible to get the best model.   
4. Including holidays in FBProphet was difficult though the data was international we have used American holidays for the model. 

## **Limitations**

- Analyzing the Impact of Single Events: When you try to assess the impact of a single event, the major problem is that there are always many events occurring at any one time. Suppose you want to find out why sales are peaking  in the March for all the four years. But there are several events happening at the same time. 
- Missing values need to be filled.
- Past performance is no indication of future results there are numrous factors that might affect the sales of the store. We can't decide the future based on just patterns. The retail market is so dynamic and often affected by new products, promotions, seasonality and other changes that make it very hard to base forward-looking decisions on past behavior. This is the reason why personal experience, local knowledge, and expert judgement are often used as critical inputs to override automated forecasting. 
