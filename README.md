# Time-Series-Analysis
Cleaning high multi-dimensional panel data, EDA, Iterative PCA on high dimensional stock market

# Overview

Cochrane in 2011 explains the average US stock returns‚Äô behavior by
identifying stock level characteristics that contain independent information.
Green et al. (2017) take Cochrane‚Äôs challenge using 94 stock level
characteristics to identify significant predictors statistically.
They use the Fama MacBeth regression, in which they include 94
characteristics from 1980 to 2014. They mainly focus on the OLS method
because its assumptions can significantly identify the most potent
independent determinants after accounting for all other factors.

# Goal

In this project, I will be developing an unsupervised technique (PCA model) to reduce the dimensionality of the data by identifying latent variables that explain patterns in the 94 firm characteristics.

# Challenge


Applying Principal Component Analysis (PCA) to high-dimensional time series data is challenging and complex due to several inherent properties and practical considerations of time series and high-dimensional datasets. Here are the key challenges:

1. Temporal Dependency Sequential Nature: Time series data points are temporally ordered and often exhibit dependencies, meaning that current values depend on previous values. Autocorrelation: The presence of autocorrelation (correlation of the time series with lagged versions of itself) complicates the assumption of independence among data points, which PCA relies on.

2. Non-Stationarity Changing Statistics: Time series data often exhibit non-stationarity, where statistical properties like mean and variance change over time due to trends, seasonality, and other factors. Trend and Seasonality: These components must be removed or accounted for before applying PCA, requiring additional preprocessing steps like differencing, detrending, or seasonal decomposition.

3. High Dimensionality Curse of Dimensionality: High-dimensional data can lead to computational inefficiencies and numerical instability in PCA calculations. Sparsity: As the number of dimensions increases, the data becomes sparse, making it harder to capture meaningful patterns. Interpretability: Interpreting principal components in a high-dimensional space can be challenging and less intuitive.

4. Feature Extraction Lagged Features: Creating lagged versions of the time series (e.g., using past values as features) increases the dimensionality further, complicating the analysis. Windowing: Dividing the time series into overlapping windows to capture local patterns introduces additional complexity in maintaining the temporal context.

5. Scalability Computational Resources: Performing PCA on large high-dimensional datasets requires significant computational resources and memory. Efficiency: Efficient algorithms and data structures are needed to handle the computation of covariance matrices and eigen decomposition in high dimensions.

6. Handling Missing Data Imputation: Time series data often contains missing values, which need to be imputed accurately to avoid biasing the PCA results. Temporal Gaps: Temporal gaps and irregular time intervals pose additional challenges for consistent data preprocessing.

7. Multivariate Time Series Interdependencies: In multivariate time series, different time series may be interdependent, adding complexity to capturing these relationships with PCA. Dimension Explosion: Combining multiple time series into a single dataset further increases dimensionality and computational burden.

# Data Overview

The data sample covers 60 years, starting in March 1957, ending December
2016. They include a total of 6,200 in each month with individual returns
obtained from CRSP for all NYSE, NASDAQ, and AMEX firms. The data
set also contains stocks with share codes beyond 10 and 11, prices below $5,
and financial firms. One reason for selecting the largest pool of assets is that one can avoid overfitting by increasing the number of observations. The
stock-level predictive characteristics consist of 20 monthly, 13 quarterly, and
61 annually updated features that are constructed based on Green et al. (2017)
with some minor modifications.

The original data contains 869,347 observations for 97 variables. Ninety-four variables correspond to
the stock-level characteristics. The two variables represent DATE and Permno, respectively. 

**Permno:** is a unique identifier for one particular stock in the database

**DATE:** is of the form yy-mm-dd.

<u>**Note:**</u> This data is huge, therefore, for the sake of representation, I am going to extract data from 2005 till 2016

## EDA - Exaplanatory Data Analysis

### Imports


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
import plotly.express as px
import datetime as dt
from collections import Counter
```

### Discover the data set 

Here's a detailed description of the variables, which are commonly used in financial and economic research, particularly in the context of firm characteristics and stock market analysis:

1. **mvel1**: Market value of equity lagged by one month.
2. **beta**: Measure of a stock's volatility relative to the overall market.
3. **betasq**: Square of the beta value.
4. **chmom**: Change in momentum; typically calculates the difference in momentum over two periods.
5. **dolvol**: Dollar volume; total value of shares traded over a period.
6. **idiovol**: Idiosyncratic volatility; stock-specific risk independent of overall market movement.
7. **indmom**: Industry momentum; average return of stocks within the same industry.
8. **mom1m**: One-month price momentum.
9. **mom6m**: Six-month price momentum.
10. **mom12m**: Twelve-month price momentum.
11. **mom36m**: Thirty-six month price momentum.
12. **pricedelay**: Measures the lag in a stock‚Äôs price adjustment to information.
13. **turn**: Turnover; ratio of the volume of shares traded to the number of shares available.
14. **absacc**: Absolute value of accruals, indicating the quality of earnings.
15. **acc**: Accruals component of earnings.
16. **age**: Age of the firm.
17. **agr**: Asset growth rate.
18. **bm**: Book-to-market ratio.
19. **bm_ia**: Industry-adjusted book-to-market ratio.
20. **cashdebt**: Ratio of cash holdings to debt.
21. **cashpr**: Cash profitability; cash flow to price ratio.
22. **cfp**: Cash flow to price ratio.
23. **cfp_ia**: Industry-adjusted cash flow to price ratio.
24. **chatoia**: Change in asset turnover, industry adjusted.
25. **chcsho**: Change in the number of shares outstanding.
26. **chempia**: Change in employee efficiency, industry adjusted.
27. **chinv**: Change in inventory levels.
28. **chpmia**: Change in profit margin, industry adjusted.
29. **convind**: Convertible indicator; marks if the firm has convertible securities.
30. **currat**: Current ratio; a liquidity ratio that measures a company‚Äôs ability to pay short-term obligations.
31. **depr**: Depreciation rate relative to total assets.
32. **divi**: Dividend initiation.
33. **divo**: Dividend omission.
34. **dy**: Dividend yield.
35. **egr**: Earnings growth rate.
36. **ep**: Earnings to price ratio.
37. **gma**: Gross margin.
38. **grcapx**: Growth in capital expenditures.
39. **grltnoa**: Growth in long-term net operating assets.
40. **herf**: Herfindahl index, a measure of the size of firms in relation to the industry and an indicator of the amount of competition among them.
41. **hire**: Hiring rate.
42. **invest**: Investment rate.
43. **lev**: Leverage ratio.
44. **lgr**: Long-term growth rate.
45. **mve_ia**: Market value of equity, industry adjusted.
46. **operprof**: Operating profitability.
47. **orgcap**: Organizational capital.
48. **pchcapx_ia**: Percentage change in capital expenditures, industry adjusted.
49. **pchcurrat**: Percentage change in current ratio.
50. **pchdepr**: Percentage change in depreciation.
51. **pchgm_pchsale**: Changes in gross margin relative to changes in sales.
52. **pchquick**: Percentage change in quick ratio.
53. **pchsale_pchinvt**: Sales growth minus inventory growth.
54. **pchsale_pchrect**: Sales growth minus receivables growth.
55. **pchsale_pchxsga**: Sales growth minus growth in SG&A expenses.
56. **pchsaleinv**: Sales growth relative to inventory growth.
57. **pctacc**: Percentage accruals.
58. **ps**: Price to sales ratio.
59. **quick**: Quick ratio, a measure of a company‚Äôs ability to meet its short-term obligations with its most liquid assets.
60. **rd**: Research and development expenses.
61. **rd_mve**: R&D to market value ratio.
62. **rd_sale**: R&D to sales ratio.
63. **realestate**: Proportion of assets in real estate.
64. **roic**: Return on invested capital.
65. **salecash**: Sales to cash ratio.
66. **saleinv**: Sales to inventory ratio.
67. **salerec**: Sales to receivables ratio.
68. **secured**: Secured debts.
69. **securedind**: Secured debt indicator.
70. **sgr**: Sales growth rate.


71. **sin**: Sin stock indicator (typically for industries like alcohol, tobacco, gambling).
72. **sp**: Sales to price ratio.
73. **tang**: Tangibility; proportion of physical assets relative to total assets.
74. **tb**: Tax burden.
75. **aeavol**: Abnormal earnings volatility.
76. **cash**: Cash holdings.
77. **chtx**: Corporate tax rate.
78. **cinvest**: Corporate investment.
79. **ear**: Earnings announcement returns.
80. **nincr**: Number of consecutive years of earnings increases.
81. **roaq**: Return on average equity.
82. **roavol**: Return on assets volatility.
83. **roeq**: Return on equity.
84. **rsup**: Relative supply; measures stock availability in the market.
85. **stdacc**: Standard deviation of accruals.
86. **stdcf**: Standard deviation of cash flows.
87. **ms**: Mass sentiment; measures market sentiment.
88. **baspread**: Bid-ask spread.
89. **ill**: Illiquidity measure.
90. **maxret**: Maximum return.
91. **retvol**: Return volatility.
92. **std_dolvol**: Standard deviation of dollar volume.
93. **std_turn**: Standard deviation of turnover.
94. **zerotrade**: Days with zero trades.
95. **sic2**: The first two digits of the Standard Industrial Classification code.


### Filter the data from 2005 to 2016
As I mentioned before, I am going to extract only a portion from the data due to its high dimensionality. The extraction process will be from january of 2005 to december of 2016.


The first data point for the year 2005 is: **20050131**
Therefore, I am going to subset the data to account for all data point beyond 2005-01-31. Call the new data frame *df1*

### Dealing with dates

```python
# Assuming 'DATE' is the datetime column
# Convert datetime to string and extract the microseconds part which contains the actual date
df2['custom_date'] = df2['DATE'].astype(str).str[-8:]

# Now convert this 'custom_date' to a proper datetime
df2['parsed_date'] = pd.to_datetime(df2['custom_date'], format='%Y%m%d')

# Check the results
print(df2[['DATE', 'custom_date', 'parsed_date']].head())
```

```python
#Create the final date column which takes the correct date format without timestamp
df2['date'] = df2['parsed_date'].dt.date
```

```python
#Drop the DATE, custom date, and parsed date columns
columns_to_drop = ['DATE','custom_date','parsed_date']
df2.drop(columns=columns_to_drop,inplace=True)
```

I have to dynamically reorder columns (e.g., move a specific column to the first position without hardcoding all column names), I can use the following approach:


```python
#Re-order the created date column
# Check if 'date' is in the columns
if 'date' in df2.columns:
    # Column to move
    column_to_move = 'date'

    # Create a new order for the columns, moving 'date' to the first position
    new_order = [column_to_move] + [col for col in df2.columns if col != column_to_move]

    # Apply the new order
    df2 = df2[new_order]

    print("Dynamically Reordered DataFrame:")
    print(df2)
else:
    print("Column 'date' not found in DataFrame df2.")

```

### Investigating data types

It looks like all the variables have reasonable data types, However, I will only transform the date column from **object** to **datetime object**


```python
# Convert 'date' column to datetime
df2['date'] = pd.to_datetime(df2['date'], errors='coerce')

# Check the changes
print(df2['date'].dtype)
```

### Dealing with missing values

```python
print(df3.isna().sum())
```

It looks like the dataframe contains a significant amount of missing values, therefore, I will investigate the missing values and decide on whether to simply remove them or perform certain imputation process.

#### Create some visualizations to better understand

##### Create a visualization for the COUNT of missing values


##### Create a Visualization for the percentage of missing values from the total dataset


##### Create a Visualization using plotly 
The Plotly library is a powerful, interactive graphing library that lets you create high-quality, interactive, and aesthetically pleasing graphs directly from your data. It supports a wide range of static, animated, and interactive visualizations and is particularly suited for web and interactive applications. Plotly is available for Python, R, and JavaScript.


### Missing Values Imputation

Since Principal Component Analysis (PCA) is sensitive to missing values and given that I visualize the missing values which are significant ratio of total data points, I have to figure out a way to impute missing values.

Imputing missing values in cross-sectional stock market data can be quite challenging due to the nature of the data, which often contains variables like stock prices, volumes, financial ratios, etc. The choice of imputation method can significantly affect the results of any subsequent analyses, such as portfolio optimization, risk assessment, or trading algorithms. The best method often depends on the specific characteristics of your data and the goals of your analysis.

Here are some common techniques for imputing missing values in stock market data, along with considerations for each:

* **1. Mean or Median Imputation:** This method is simple and can be effective if the data is roughly normally distributed without outliers.

    Pros: Easy to implement and understand.
    
    Cons: Can reduce the variance of the dataset and is not suitable for data with trends or seasonality.
    

* **2. Last Observation Carried Forward (LOCF):** Useful for time-series data where the most recent observation might be the best guess for the next value, such as stock prices or indices.

    Pros: Maintains continuity in data that is sequentially dependent.
    
    Cons: Can introduce bias if the previous value is not representative of future values, especially over longer gaps.
    
    
* **3. Linear Interpolation:** Effective for time-series data where values are expected to change incrementally over time.

    Pros: Provides a smooth estimate of missing values based on surrounding data points.
    
    Cons: May not be appropriate for volatile stock data where changes are not gradual.
    
    
* **4. Regression Imputation:** When other related variables (predictors) can be used to estimate the missing values. For example, using sector indices to estimate missing individual stock prices.

    Pros: Can provide accurate imputations if the relationships between variables are strong and well understood.
    
    Cons: Risk of introducing bias if the model is misspecified; computationally intensive.
    
    
* **5. Multiple Imputation:** Considered a more sophisticated approach to handle missing data by creating multiple complete datasets using a stochastic (random) method, analyzing each dataset separately, and then pooling the results.

    Pros: Accounts for the uncertainty in the imputations and typically provides more accurate inference than single imputation methods.
    
    Cons: More complex to implement and interpret; requires statistical software capable of performing multiple imputation and combining results.
    
    
* **6. Advanced Machine Learning Methods (e.g., K-Nearest Neighbors, Deep Learning)** When you have large datasets and computational resources to support more complex imputation models. These methods can consider non-linear relationships and interactions between features.

    Pros: Can capture complex patterns in the data, potentially leading to more accurate imputation.
    
    Cons: Requires significant computational resources; risk of overfitting; complex models can be difficult to interpret.


#### Checking for data distribution

It looks like some of the variables are normally distributed and some are skewed. **Therefore**, I decided to impute missing values using <u>**cross-sectional median**</u>

#### Why Cross-sectional Median Imputation


Imputing missing values using the cross-sectional median in a panel stock market data offers several advantages:

* Robustness to Outliers:The median is less sensitive to extreme values compared to the mean. Stock market data often contains outliers due to sudden market movements, and using the median helps to mitigate the influence of these anomalies.
  
* Simple and Intuitive:The median is straightforward to calculate and understand, making it an accessible method for handling missing data without requiring complex algorithms or assumptions.
  
* Preservation of Data Distribution:Using the median helps maintain the central tendency of the data, ensuring that the imputed values are representative of the typical values within the cross-section at each time point.
  
* Effective for Skewed Data:Stock market data can be highly skewed. The median is a better measure of central tendency for skewed distributions, providing a more accurate reflection of the data's central value.
  
* Consistency Across Time:In panel data, consistency in imputation methods is crucial. Using the cross-sectional median ensures that the method is uniformly applied across different time points, enhancing the reliability of the imputed values.
  
* Reduction of Bias:Imputing missing values with the cross-sectional median helps reduce bias in the data. It avoids over-representing or under-representing certain values that might occur if mean imputation were used in the presence of skewed distributions or outliers.
  
* Preservation of Variability:While the median is a measure of central tendency, it does not significantly distort the variability within the data. This helps in preserving the inherent characteristics and variability of the stock market data.
  
* Ease of Implementation:Median imputation is computationally efficient and easy to implement, making it a practical choice for large datasets, such as those often encountered in stock market analyses.

#### Impute Missing Values Using the Computed Medians
Grouping and Transform: Instead of grouping and then applying a function row by row, transform is used. This method applies a function to each group and then returns a DataFrame having the same indices as the original, which makes it perfect for filling NA values directly.

Fill NA with Medians: The fillna() method is inherently vectorized and when passed the DataFrame of medians (aligned with the original DataFrame), it fills the missing values efficiently.



```python
# Calculate the median for each date
medians = df3[df3.columns.difference(['permno', 'date'])].groupby(df3['date']).transform('median')
#This line is used to select all columns except permno and the date column
#(which typically wouldn't be imputed or included in median calculations).

# Fill missing values with medians obtained (using vectorization)
df_filled = df3.fillna(medians)

print("DataFrame after efficient median imputation:")
print(df_filled)
```


```python
#Check if all missing values were imputed
df_filled.isna().sum()
```


```python
df_filled.head()
```


```python
#Double check if the shape of original data frame is similar to the cleaned data frame
print(df_filled.shape)
print(df1.shape)
```

#### Save the final cleaned data frame 


```python
#df_filled.to_csv('Cleaned_data1_2005-2016.csv', index=False)  # 'index=False' omits the index from the filed
```

## PCA applied on Multi-dimensional Time Series Data

### Definition

Principal Component Analysis (PCA) is a statistical technique used for dimensionality reduction. It transforms a dataset with potentially correlated variables into a set of linearly uncorrelated variables called principal components. The primary goal of PCA is to reduce the number of variables (dimensions) in the data while preserving as much variance (information) as possible. Here‚Äôs a detailed definition and explanation:

### Key Concepts

* 1- Dimensionality Reduction: PCA reduces the number of variables in a dataset by transforming it into a new set of variables, the principal components, which are fewer in number and uncorrelated.
  
* 2- Principal Components: These are new variables constructed as linear combinations of the original variables.
Each principal component captures the maximum possible variance in the data, subject to the constraint that it is orthogonal to (uncorrelated with) the preceding components.

* 3- Variance Maximization: PCA aims to capture the maximum amount of variance with the fewest number of principal components.
The first principal component accounts for the largest possible variance, the second principal component (orthogonal to the first) accounts for the next largest variance, and so on.

* 4- Orthogonality: Principal components are orthogonal (uncorrelated) to each other, ensuring that each component captures unique information from the dataset.

### Steps in PCA

* 1- Standardization: If the variables have different units or scales, standardize the data to have a mean of zero and a standard deviation of one.
  
* 2- Covariance Matrix Computation: Compute the covariance matrix to understand how the variables in the dataset vary with respect to each other.
  
* 3- Eigenvalues and Eigenvectors: Calculate the eigenvalues and eigenvectors of the covariance matrix. The eigenvectors determine the direction of the principal components, and the eigenvalues determine their magnitude (amount of variance they capture).
  
* 4- Principal Components Calculation: Sort the eigenvalues and corresponding eigenvectors in descending order of the eigenvalues. The top eigenvectors become the principal components.
  
* 5- Transform Data: Project the original data onto the principal components to get the transformed dataset.

### Application of PCA

* 1- Data Visualization: Reduce high-dimensional data to 2 or 3 principal components for easy visualization.
 
* 2- Noise Reduction: Remove noise and redundant information from data by keeping only the most significant components.

* 3- Feature Extraction and Engineering: Generate new features that capture the most important information from the original data.

* 4- Preprocessing for Machine Learning: Simplify models by reducing the number of input variables, leading to faster training and potentially better performance by eliminating irrelevant features.

### Project Goal and the Challenge

Applying Principal Component Analysis (PCA) to high-dimensional time series data is challenging and complex due to several inherent properties and practical considerations of time series and high-dimensional datasets. Here are the key challenges:

**1. Temporal Dependency**
Sequential Nature: Time series data points are temporally ordered and often exhibit dependencies, meaning that current values depend on previous values.
Autocorrelation: The presence of autocorrelation (correlation of the time series with lagged versions of itself) complicates the assumption of independence among data points, which PCA relies on.

**2. Non-Stationarity**
Changing Statistics: Time series data often exhibit non-stationarity, where statistical properties like mean and variance change over time due to trends, seasonality, and other factors.
Trend and Seasonality: These components must be removed or accounted for before applying PCA, requiring additional preprocessing steps like differencing, detrending, or seasonal decomposition.

**3. High Dimensionality**
Curse of Dimensionality: High-dimensional data can lead to computational inefficiencies and numerical instability in PCA calculations.
Sparsity: As the number of dimensions increases, the data becomes sparse, making it harder to capture meaningful patterns.
Interpretability: Interpreting principal components in a high-dimensional space can be challenging and less intuitive.

**4. Feature Extraction**
Lagged Features: Creating lagged versions of the time series (e.g., using past values as features) increases the dimensionality further, complicating the analysis.
Windowing: Dividing the time series into overlapping windows to capture local patterns introduces additional complexity in maintaining the temporal context.

**5. Scalability**
Computational Resources: Performing PCA on large high-dimensional datasets requires significant computational resources and memory.
Efficiency: Efficient algorithms and data structures are needed to handle the computation of covariance matrices and eigen decomposition in high dimensions.

**6. Handling Missing Data**
Imputation: Time series data often contains missing values, which need to be imputed accurately to avoid biasing the PCA results.
Temporal Gaps: Temporal gaps and irregular time intervals pose additional challenges for consistent data preprocessing.

**7. Multivariate Time Series**
Interdependencies: In multivariate time series, different time series may be interdependent, adding complexity to capturing these relationships with PCA.
Dimension Explosion: Combining multiple time series into a single dataset further increases dimensionality and computational burden.

### Notes

Note the following:

* 1- An extensive EDA has been applied to this data in another project
* 2- Missing values were imputed in another project using cross-sectional median imputation
* 3- Time series analysis was applied in another project were I studied the stationarity and linearity of the variables
* 4- Statistical tests (ADF and KPSS) were applied before do investigate stationarity of time series

* **MOST Importantly:** The aim of the following PCA anlysis is to just extract the most informative feauters out of 95 stock level characteristics to reduce the dimensionality of the data set so I can use it later to develop advanced machine learning techniques on the US stock market data.

#### Imports


```python
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


import matplotlib.pyplot as plt
import warnings

import datetime as dt
import os

from collections import Counter
```


```python
df = pd.read_csv('/Users/adel/Desktop/Masters Thesis/Cleaned_data1_2005-2016.csv')
```

#### Subset the data to take all stock ids that are available in all years


```python
# Ensure the date column is of datetime type
df['date'] = pd.to_datetime(df['date'])

# Step 1: Identify all unique dates in the dataset
unique_dates = df['date'].nunique()

# Step 2: Group by permno and count the unique dates for each permno
permno_date_counts = df.groupby('permno')['date'].nunique()

# Step 3: Filter permno values that are present in all unique dates
permno_in_all_dates = permno_date_counts[permno_date_counts == unique_dates].index

# Filter the original DataFrame to include only rows with these permno values (if needed)
df_filtered = df[df['permno'].isin(permno_in_all_dates)]
```

#### Filter the data to take only year of 2015 and 2016


```python
df1 = df_filtered[df_filtered['date'] >= '2015-01-30']
```

#### Randomly select 50 stock ids to perform the analysis on 


```python
# Ensure the permno column has the desired number of unique permno values
unique_permno = df1['permno'].unique()

# Check the number of unique permno values
total_unique_permno = len(unique_permno)
print(f"Total unique permno values: {total_unique_permno}")

# Define the number of permno values you want to subset
num_permno_to_select = 50

# Set the random seed for reproducibility
np.random.seed(40)

# Randomly select the desired number of permno values
selected_permno = np.random.choice(unique_permno, num_permno_to_select, replace=False)

# Filter the original DataFrame to include only rows with these selected permno values
df_50 = df1[df1['permno'].isin(selected_permno)]
```

#### Create a nested for loop to iterate over every permno for each year and fit a PCA model 

By creating a nested for loop that iterates on both a unique stock ID and a unique date, this will take into consideration that PCA model is being applied not only on a time series data but on a Panel Data (Time Series Data + Cross-sectional Data)

**Limitations:** It has an increased computation cost since its fitting a pca model on every stock ID and specific year 

**advantages:** This model is optimal in working with high dimensional panel data and provides acurate results that accomedates for temperoral changes in a time series 


```python
# List to store PCA results
pca_results = []

# Get unique stock IDs and years
unique_stock_ids = df_50_f['permno'].unique()
unique_years = df_50_f['year'].unique()

# Iterate over each stock ID and year combination
for stock_id in unique_stock_ids:
    for year in unique_years:
        # Print current stock_id and year to track progress
        print(f"Processing stock_id: {stock_id}, year: {year}")
        print("________________________________________________________")
        # Filter the DataFrame for the current stock ID and year
        subset_df = df_50_f[(df_50_f['permno'] == stock_id) & (df_50_f['date'].dt.year == year)]
        print("Subset data frame", subset_df)
        print("_________________________________________________________")

        # Ensure we're working with a copy
        subset_df = subset_df.copy()
        
        # Check if the subset is not empty and has more than one sample
        if not subset_df.empty and len(subset_df) > 1:
            # Drop non-feature columns
            features_df = subset_df.drop(columns=['date', 'permno','year'], axis=1)
            print("-----------------------------------------------------")
            print("Features dataframe", features_df)
           
            # Initialize the scaler and standardize the data   
            scaler = StandardScaler()
            standardized_features = scaler.fit_transform(features_df)
            print("------------------------------------------------------")
            print("Standardized features:", standardized_features)
                  
            # Determine the number of components dynamically
            n_components = min(len(features_df), features_df.shape[1])

            # Apply PCA
            pca = PCA(n_components=n_components)
            principal_components = pca.fit_transform(standardized_features)
            print("Principal Components:", principal_components)

            # Store the results
            pca_result = {
                'stock_id': stock_id,
                'year': year,
                'principal_components': principal_components,
                'explained_variance': pca.explained_variance_ratio_,
                'loadings': pca.components_.T
            }
            pca_results.append(pca_result)
            
            print("------------------------------------------------------")
            print(pca_result)
            print("------------------------------------------------------")
```

#### Store the PC's and loadings into a dataframe 


```python
# Lists to store the principal components and loadings dataframes
principal_dfs = []
loadings_dfs = []

# Iterate over all PCA results
for i, result in enumerate(pca_results):
    print(f"Processing PCA result {i+1}")
    
    print("Stock ID:", result['stock_id'])
    print("Year:",result['year'])
    print("Explained Variance Ratios:", result['explained_variance'])
    print("Principal Components:\n", result['principal_components'])
    print("Loadings (Eigenvectors):\n", result['loadings'])


    # Save the results into a dataframe
    principal_df = pd.DataFrame(data=result['principal_components'])
    
    # Save the loadings into a dataframe
    loadings_df = pd.DataFrame(data=result['loadings'], columns=[f'PC{j+1}' for j in range(result['loadings'].shape[1])],
                               index=features_df.columns)


    # Append the dataframes to the lists
    principal_dfs.append(principal_df)
    loadings_dfs.append(loadings_df)
```

#### Extract the top 20 informative feautures


```python
# Define the stock_id and year arrays
stock_ids = np.array([10044, 11581, 11786, 18403, 20117, 24969, 52090, 53225, 61508,
       62958, 63125, 63765, 64629, 70033, 72996, 75039, 76123, 76804,
       77768, 78629, 79864, 80233, 81044, 81084, 81740, 82276, 82541,
       83856, 83976, 84032, 84397, 85058, 85346, 85839, 87379, 87447,
       87505, 87541, 88466, 88615, 88742, 89365, 89591, 89708, 89746,
       89879, 89897, 89965, 90266, 90346])

years = np.array([2015, 2016])

# Create the list of dictionaries
metadata = [{"stock_id": stock_id, "year": year} for stock_id in stock_ids for year in years]

# Print the list of dictionaries
for entry in metadata:
    print(entry)
```


```python
# Directory to save the plots
#save_dir = 'pca_plots'
#os.makedirs(save_dir, exist_ok=True)
```


```python
def top_features(loadings, stock_id, year):
    # Compute the absolute loadings
    absolute_loadings = loadings.abs()

    # Sum the absolute loadings for each variable across all components
    variable_importance = absolute_loadings.sum(axis=1)

    # Sort variables by importance
    sorted_importance = variable_importance.sort_values(ascending=False)

    print("Variable Importance:")
    print(sorted_importance)

    # Select the top 20 most important variables
    top_20_variables = sorted_importance.head(20)

    print("Top 20 Most Important Variables:")
    print(top_20_variables)

    # Plot the top features
    plt.figure(figsize=(10, 6))
    plt.barh(top_20_variables.index, top_20_variables.values, color='skyblue')
    plt.xlabel('Importance')
    plt.title(f'Top Informative Features for Stock ID: {stock_id} (Year: {year})')
    plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature on top

    # Ensure the directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Save the plot
    plot_path = os.path.join(save_dir, f'top_features_{stock_id}_{year}.png')
    plt.savefig(plot_path)
    plt.close()


    return top_20_variables.index.tolist()
    
    
# Counter to store the occurrences of each variable
variable_counter = Counter()

# Directory to save plots
save_dir = '50_permno_pca'

# Iterate over each DataFrame in loadings_dfs and apply the top_features function
for i, loadings_df in enumerate(loadings_dfs):
    print(f"Processing DataFrame {i+1}")
    stock_id = metadata[i]['stock_id']
    year = metadata[i]['year']
    top_20_variables = top_features(loadings_df, stock_id, year)
    
    # Count the occurrences of each variable
    variable_counter.update(top_20_variables)

# Identify the most occurring variables
most_common_variables = variable_counter.most_common(20)

# Create a DataFrame to store the common variables
common_variables_df = pd.DataFrame(most_common_variables, columns=['Variable', 'Count'])

# Print the DataFrame
print("Common Variables DataFrame:")
print(common_variables_df)
```

### Conclusion

The following PCA model was iterated for 50 stock ids for the year 2015 and 2016. 

Reason: The reason behind this iterative PCA model is to account for the high-dimensional time series in which there is a unique pca applied on every stock at each year for optimal results. 
In addition, this will help me identify thee best stock features for a unique stock for further analysis.

After finalizing the model, I extracted the most occuring features for all stocks to have a general idea on what stock features are important in analyzing stock market data (Fundemantal Analysis).

The top 20 informative stock level characteritics ranked in order are:

0) mom1m    
1) zerotrade     
2) dolvol     
3) pricedelay     
4) std_dolvol     
5) maxret     
6) ill     
7) mvel1     
8) std_turn     
9) baspread     
10) retvol     
11) beta     
12) chmom     
13) betasq     
14) mom6m     
15) indmom     
16) mom36m     
17) mom12m     
18) idiovol     
19) turn     


```python

```

![image](https://github.com/AdelMalaeb/Time-Series-Analysis/assets/166052475/b40ff76d-8087-44bc-a495-4d61c252467c)
