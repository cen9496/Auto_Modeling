# Data Modeling Tools
Author: Christopher Napier <cenapier9496@gmail.com>

For questions, comments, or suggestions, please feel free to email me.

For futher understanding of the individual functions, feel free to analyze the modules provided which have further details.

## Overview

The goal of this package is to help create automated processes for building models. It leverages the idea that you can find a "best model" by maximizing things such as R2 or minimizing AIC/BIC. While some practitioners only believe in these comparisions for nested models, this package performs the comparison against all possible combinations of parameters, including non-nested models.

To this end, comparisons like the likelihood ratio test are not currently implemented as it relies on nested model comparison. This project may expand to allow "best" model creation leveraging those ideas, but currently the package is built to solely maximize/minimize specific measures for fit.

## Metrics

Currently, there are a host of metrics that we can obtain utilizing this package. Below will be a introduction on how to call them. For future refrence, you may want to look up what the metrics are for yourself for a more in depth understanding.

### Input Definitions

* **model** - The Model you want to input
* **x_train** - The input data for the parameters you want to use for prediction, remove the parameter you want to predict. Do not include extra parameters that aren't being used in the model.
* **y_train** - The data you want to to predict, i.e. your output data\actual values. Ensure it is only the parameter you want to predict.
* **num_groups** - The number of groups you want to break the data into.

### get_metrics(model, x_train, y_train)

This function automatically generates metrics that are relevant to the modle you're utilizing. The goal is to allow for less coding to generate measures commonly used for building models and this function is leveraged by the automatic modeling building funcitons later.

For example, for Logistical Regression, it will generate things like the McFadden R-Squared measure or the Hosmer-Lemeshow information. Whereas for Linear Regression, you will get the Mean Squared Error (MSE) and the Mean Absolute Error (MAE). These are just a few examples as it will generate more.

This function also outputs the pvalues for the parameters, so the user can decide if they're significant enough to leave in or remove.

Below is an example of how to get the values returned by this function and print them:

```python
metrics, pvalues = get_metrics(model, x_train, y_train)
print(metrics)
print(pvalues)
```

### get_pvalues(model, x_train, y_train)

This function returns the P-Values associated with the individual parameters used in the model. This allows the user to decide if the parameters are significant enough to use in the model.

```python
pvalues = get_pvalues(model, x_train, y_train)
print(pvalues)
```

### get_log_lik(model, x_train, y_train)

This function returns the Log Likelihood of the model based on the data. It currently returns the likelihood for both Logistic and Linear Regression models.

Below is an example of how to get the values returned by this function:

```python
Log_Likelihood = get_log_lik(model, x_train, y_train)
```

### get_aic(model, x_train, y_train)

This function returns the AIC (Akaike Information Criterion) of the model based on the data. It currently returns the AIC for both Logistic and Linear Regression models.

Below is an example of how to get the values returned by this function:

```python
AIC = get_aic(model, x_train, y_train)
```

### get_bic(model, x_train, y_train)

This function returns the BIC (Bayesian Information Criterion) of the model based on the data. It currently returns the BIC for both Logistic and Linear Regression models.

Below is an example of how to get the values returned by this function:

```python
BIC = get_bic(model, x_train, y_train)
```

### get_null_deviance(model, x_train, y_train)

This function returns the Null Deviance of the null model. This is currently only relevant for Logistical Regression. The null model is the model where only the intercept is used for prediction and all the parameters have estimates (coeficients) of 0.

Below is an example of how to get the values returned by this function:

```python
Null_Dev = get_null_deviance(model, x_train, y_train)
```

### get_null_df(model, x_train, y_train)

This function returns the Degrees of Freedom for the null model.

Below is an example of how to get the values returned by this function:

```python
Null_DF = get_null_df(model, x_train, y_train)
```

### get_residual_deviance(model, x_train, y_train)

This function returns the Residual Deviance of the model. For Linear Regression, this is also known as the Sun of Square Errors (SSE).

Below is an example of how to get the values returned by this function:

```python
Resid_Dev = get_residual_deviance(model, x_train, y_train)
```

### get_residual_df(model, x_train, y_train)

This function returns the Degrees of Freedom for the null model.

Below is an example of how to get the values returned by this function:

```python
Resid_DF = get_residual_df(model, x_train, y_train)
```

### get_R2(model, x_train, y_train)

This function returns the R2 value for Linear Regression models.

Below is an example of how to get the values returned by this function:

```python
R2 = get_R2(model, x_train, y_train)
```

### get_Adj_R2(model, x_train, y_train)

This function returns the Adjusted R2 value for Linear Regression models.

Below is an example of how to get the values returned by this function:

```python
Adj_R2 = get_Adj_R2(model, x_train, y_train)
```

### get_Sigma(model, x_train, y_train)

This function returns the Residual Standard Error for Linear Regression models.

Below is an example of how to get the values returned by this function:

```python
Sigma = get_Sigma(model, x_train, y_train)
```

### get_F_Statistic(model, x_train, y_train)

This function returns the F-Statistic vs the null model to help determine if the model is significant over the null model.

Below is an example of how to get the values returned by this function:

```python
F_Stat = get_F_Statistic(model, x_train, y_train)
```

### get_F_Stat_P_Value(model, x_train, y_train)

This function returns the P-Value corresponding to the F-Statistic vs the null model, which allows the user to determine if the model is significant or not.

Below is an example of how to get the values returned by this function:

```python

F_Stat_P = get_F_Stat_P_Value(model, x_train, y_train)
```

### get_MSE(model, x_train, y_train)

This function returns the MSE (Mean Square Error) value for Linear Regression models.

Below is an example of how to get the values returned by this function:

```python
MSE = get_MSE(model, x_train, y_train)
```

### get_RMSE(model, x_train, y_train)

This function returns the RMSE (Root MSE) value for Linear Regression models.

Below is an example of how to get the values returned by this function:

```python
RMSE = get_RMSE(model, x_train, y_train)
```

### get_MAE(model, x_train, y_train)

This function returns the MAE (Mean Absolute Error) value for Linear Regression models.

Below is an example of how to get the values returned by this function:

```python
MAE = get_MAE(model, x_train, y_train)
```

### get_Pearson_Chi_Square(x_train, y_train)

This function performs the Pearson Chi Square test on all the parameters you plan to use in the model. This is a good test for significance when it comes to using classification parameters to predict classificaion outputs, such as Logistical Regression.

This will return P-Values for continuous variables as well. So, it is important to know your data as using the Pearson Chi Square test is not a good measure for continuous parameters used in classification models.

Below is an example of how to get the values returned by this function:

```python
PChiSq = get_Pearson_Chi_Square(x_train, y_train)
```

### get_McFadden_R2(model, x_train, y_train)

This function returns the McFadden R2 value for Logistic Regression models.

Below is an example of how to get the values returned by this function:

```python
McFadden_R2 = get_McFadden_R2(model, x_train, y_train)
```

### get_McFadden_adj_R2(model, x_train, y_train)

This function returns the McFadden Adjusted R2 value for Logistic Regression models.

Below is an example of how to get the values returned by this function:

```python
McFadden_Adj_R2 = get_McFadden_adj_R2(model, x_train, y_train)
```

### get_Efrons_R2(model, x_train, y_train)

This function returns the Efrons R2 value for Logistic Regression models.

Below is an example of how to get the values returned by this function:

```python
Efrons_R2 = get_Efrons_R2(model, x_train, y_train)
```

### get_Cox_Snell_R2(model, x_train, y_train)

This function returns the Cox Snell R2 value for Logistic Regression models.

Below is an example of how to get the values returned by this function:

```python
Cox_Snell_R2 = get_Cox_Snell_R2(model, x_train, y_train)
```

### get_Craig_Uhler_R2(model, x_train, y_train)

This function returns the Craig Uhler R2 value for Logistic Regression models.

Below is an example of how to get the values returned by this function:

```python
Craig_Uhler_R2 = get_Craig_Uhler_R2(model, x_train, y_train)
```

### get_Accuracy(model, x_train, y_train)

This function gets the accuracy of the model. Currently, this is only used for Logistic Regression to see how many predictions are correct out of the total number of predictions.

Below is an example of how to get the values returned by this function:

```python
Accuracy = get_Accuracy(model, x_train, y_train)
```

### get_Hosmer_Lemeshow_Test(model, x_train, y_train, num_groups = 10, return_everything = False)

This function returns the results from the Hosmer Lemeshow Test. The result is a data frame that has the Chi Squared Statistic, Degrees of Freedom, and the associated P-Value.

Below is an example of how to get the values returned by this function:

```python
HLT = get_Hosmer_Lemeshow_Test(model, x_train, y_train, num_groups = 10)
```

### Confusion_Matrix(model, x_train, y_train)

This function returns the confusion matrix for the data. This allows an analysis of where the errors in the prediction occured.

Below is an example of how to get the values returned by this function:

```python
Confusion_Matrix = Confusion_Matrix(model, x_train, y_train)
```

## Model_Building

### Input Definitions

* **x_train** - The input data for the parameters you want to use for prediction, remove the parameter you want to predict. Do not include extra parameters that aren't being used in the model.
* **y_train** - The data you want to to predict, i.e. your output data\actual values. Ensure it is only the parameter you want to predict.
* **metric** - The metric you want to use to build your function.
* **iterations** - The max number of iterations you want to allow sklearn to utilize
* **print_steps** - Set to True if you want to see how the model selection as the model is updated at each step.
* **test_percentages** - A list of percentages that you want to use to break up the data into. For example, if you want to build models where 20% and 30% of the data is used for testing, you would do [0.2, 0.3] or if you just want one test you could do [0.2].

### Logistic Modeling

Currently, with Logistical Modeling, we have 2 ways to build models. The first way is the build a singular model and the second way is to build multiple models using "buckets", which breaks the input data into training and test buckets based on percentages. More on these modeling methods will be discussed below.

#### Build_Log_Model(x_train, y_train, metric, iterations = 100000, print_steps = False)

This function will automatically build a model for you based on the metric choosen. The list of metrics you can use are below, make sure to type them in exactly as below surrounded by quotes.

* **Log Likelihood** - Maximizes the Log Likelihood if choosen.
* **AIC** - Minimizes the AIC if choosen.
* **BIC** - Minimizes the BIC if choosen.
* **Residual Deviance** - Minimizes the Residual Deviance if choosen.
* **McFadden R2** - Maximizes McFadden R2 if choosen.
* **McFadden Adjusted R2** - Maximizes McFadden Adjusted R2 if choosen.
* **Efrons R2** - Maximizes Efron's R2 if choosen.
* **Cox Snell R2** - Maximizes Cox snell R2 if choosen.
* **Craig Uhler R2** - Maximizes Craig Uhler R2 if choosen.
* **Accuracy** - Maximizes the accuracy if choosen.

This function will go through all combinations of parameters in the input data and select the combination that "best" fits the metric choosen. The number of parameters used can be as few as 1 to all the parameters found in your data.

The return values of this function are a model, the metrics for the final model, the P-Values for the estimates, and the combination of parametrs used.

Examples of how to utilize this function is found below:

```python
# Minimize AIC
Model, Metrics, PVals, Parameters = Build_Log_Model(x_train, y_train, "AIC")

# Print Steps
Model, Metrics, PVals, Parameters = Build_Log_Model(x_train, y_train, "AIC", print_steps = True)

# Change max number of iterations

Model, Metrics, PVals, Parameters = Build_Log_Model(x_train, y_train, "AIC", iterations = 100)

# Get all the input data only for the parameters choosen
x_train[Parameters]

# Predict values from a x_test dataset
Model.predict(x_test)

# Get Accuracy from x_test dataset with corresponding y_test
get_Accuracy(Model, x_test, y_test)
```

#### def Build_Log_Model_Buckets(x_train, y_train, metric, test_percentages = [.2,.3,.4], iterations = 10000, print_steps = False):

This model allows the user to build "Buckets" of data to test. For example, maybe 70% of the data will be the training set and 30% will be the test set. This function allows the user to build several models at once if they want to as you can specify mulitple test percentages as a list, then use any of the models corresponding to those percentages for your prediction.

By default, this function will create models for 20%, 30%, and 40% of the data being used as the test data as represented by the list [.2, .3, .4] above. This leverages the automatic building function previously highlighted to build these models.

Examples of how to utilize this function can be found below:

```python
# The output is a dictionary for each percentage level
# Each percentage level has the 4 outpus that the
# Build_Log_Model function outputs
models = Build_Log_Model_Buckets(x_train, y_train, metric)

# Get metrics for 0.2, can be done for 0.3 and 0.4 as well
models[0.2]               # All the Model Results
model = models[0.2]["Model"]       # The Model That can be used
metrics = models[0.2]["Metrics"]   # The Metrics for the final model
pvals = models[0.2]["p.values"]    # The p values of the estimates of the final model
params = models[0.2]["Parameters"] # The parameters used in the final model

# Change test_percentages to 1 value
models = Build_Log_Model_Buckets(x_train, y_train, metric, test_percentages = [0.25])

model = models[0.25]["Model"] # Repeat similar to the above example for the rest.

# Change test_percentages to multiple values
models = Build_Log_Model_Buckets(x_train, y_train, metric, test_percentages = [0.25, 0.35, 0.47, .18])

model = models[0.18]["Model"] # Repeat similar to the above example for the rest.
```

### Linear Modeling

Currently, with Linear Modeling, we have 2 ways to build models. The first way is the build a singular model and the second way is to build multiple models using "buckets", which breaks the input data into training and test buckets based on percentages. More on these modeling methods will be discussed below.

#### Build_Linear_Model(x_train, y_train, metric, iterations = 100000, print_steps = False)

This function will automatically build a model for you based on the metric choosen. The list of metrics you can use are below, make sure to type them in exactly as below surrounded by quotes.

* **Log Likelihood** - Maximizes the Log Likelihood if choosen.
* **AIC** - Minimizes the AIC if choosen.
* **BIC** - Minimizes the BIC if choosen.
* **Residual Deviance** - Minimizes the Residual Deviance if choosen.
* **Multiple R2** - Maximizes Multiple (Standard) R2 if choosen.
* **Adj Adjusted R2** - Maximizes Adjusted R2 if choosen.
* **MSE** - Minimizes MSE if choosen.
* **RMSE** - Minimizes RMSE if choosen.
* **MAE** - Minimizes MAE if choosen.

This function will go through all combinations of parameters in the input data and select the combination that "best" fits the metric choosen. The number of parameters used can be as few as 1 to all the parameters found in your data.

The return values of this function are a model, the metrics for the final model, the P-Values for the estimates, and the combination of parametrs used.

Examples of how to utilize this function is found below:

```python
# Minimize AIC
Model, Metrics, PVals, Parameters = Build_Linear_Model(x_train, y_train, "AIC")

# Print Steps
Model, Metrics, PVals, Parameters = Build_Linear_Model(x_train, y_train, "AIC", print_steps = True)

# Change max number of iterations

Model, Metrics, PVals, Parameters = Build_Linear_Model(x_train, y_train, "AIC", iterations = 100)

# Get all the input data only for the parameters choosen
x_train[Parameters]

# Predict values from a x_test dataset
Model.predict(x_test)

# Get R2 from x_test dataset with corresponding y_test
get_R2(Model, x_test, y_test)
```

#### def Build_Linear_Model_Buckets(x_train, y_train, metric, test_percentages = [.2,.3,.4], iterations = 10000, print_steps = False):

This model allows the user to build "Buckets" of data to test. For example, maybe 70% of the data will be the training set and 30% will be the test set. This function allows the user to build several models at once if they want to as you can specify mulitple test percentages as a list, then use any of the models corresponding to those percentages for your prediction.

By default, this function will create models for 20%, 30%, and 40% of the data being used as the test data as represented by the list [.2, .3, .4] above. This leverages the automatic building function previously highlighted to build these models.

Examples of how to utilize this function can be found below:

```python
# The output is a dictionary for each percentage level
# Each percentage level has the 4 outpus that the
# Build_Linear_Model function outputs
models = Build_Linear_Model_Buckets(x_train, y_train, metric)

# Get metrics for 0.2, can be done for 0.3 and 0.4 as well
models[0.2]               # All the Model Results
model = models[0.2]["Model"]       # The Model That can be used
metrics = models[0.2]["Metrics"]   # The Metrics for the final model
pvals = models[0.2]["p.values"]    # The p values of the estimates of the final model
params = models[0.2]["Parameters"] # The parameters used in the final model

# Change test_percentages to 1 value
models = Build_Linear_Model_Buckets(x_train, y_train, metric, test_percentages = [0.25])

model = models[0.25]["Model"] # Repeat similar to the above example for the rest.

# Change test_percentages to multiple values
models = Build_Linear_Model_Buckets(x_train, y_train, metric, test_percentages = [0.25, 0.35, 0.47, .18])

model = models[0.18]["Model"] # Repeat similar to the above example for the rest.
```