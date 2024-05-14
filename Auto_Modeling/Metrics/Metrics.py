import pandas as pd
import sklearn as skl
import numpy as np
import copy
import scipy
import sklearn.feature_selection

# Automatically Generate Metrics Based on Model Type #
def get_metrics(model, x_train, y_train):

    # Get Metrics RFE - Depreciated and needs review 
    if type(model) == skl.feature_selection.RFE:
        y_hat = model.predict_log_proba(x_train)
        log_likelihood = y_hat[np.arange(len(y_train)),  y_train].sum()
        selected_features = [feature for feature, contained in zip(Potential_Features, model.support_) if contained == True]
        act_aic = 2*(len(selected_features)+1) - 2 * (log_likelihood)
        log_loss = skl.metrics.log_loss(y_train, model.predict_proba(x_train))
        log_like = -1 * log_loss
        mse = skl.metrics.mean_squared_error(y_test, model.predict(x_test))
        r2 = skl.metrics.r2_score(y_test,model.predict(x_test))
        r2adj = 1 - (1 - r2)*(len(y_test) - 1)/(len(y_test) - len(selected_features) - 1)
        mfr2 = skl.metrics.log_loss(y_test, [y_test.mean()]*len(y_test))
        aic = 2*(len(selected_features)+1) - 2 * (log_likelihood)
        metrics = pd.DataFrame.from_dict({'Log_Loss':[log_loss],'Log_Like':[log_like],'MSE':[mse],'R2':[r2],'R2ADJ':[r2adj],'AIC':[aic], "McFadden R2":[mfr2]})

    # Get Metrics for Logistic Regressions 
    elif type(model) == skl.linear_model.LogisticRegression:
        Hosmer_Lemeshow = get_Hosmer_Lemeshow_Test(model, x_train, y_train)
        metrics = pd.DataFrame.from_dict({"Measures": ["Log Likelihood", "AIC", "BIC", "Residual Deviance",
                                                       "Null Deviance", "McFadden R2",
                                                       "McFadden Adjusted R2", "Efrons R2", "Cox Snell R2",
                                                       "Craig Uhler R2", "Hosmer-Lemeshow", "Accuracy"],
                                          "Values": [get_log_lik(model, x_train, y_train), get_aic(model, x_train, y_train),
                                                     get_bic(model, x_train, y_train), get_residual_deviance(model, x_train, y_train),
                                                     get_null_deviance(model, x_train, y_train), get_McFadden_R2(model, x_train, y_train),
                                                     get_McFadden_adj_R2(model, x_train, y_train), get_Efrons_R2(model, x_train, y_train),
                                                     get_Cox_Snell_R2(model, x_train, y_train), get_Craig_Uhler_R2(model, x_train, y_train),
                                                     Hosmer_Lemeshow["Chi_Squared"].values[0], get_Accuracy(model, x_train, y_train)],
                                          "df": ["", "", "", get_residual_df(model, x_train, y_train),
                                                 get_null_df(model, x_train, y_train), "", "",
                                                 "", "", "", Hosmer_Lemeshow["df"].values[0], ""],
                                          "p.value": ["", "", "", "",
                                                      "", "", "", "",
                                                      "", "",
                                                      Hosmer_Lemeshow["p.value"].values[0], ""]})
        return metrics, get_pvalues(model, x_train, y_train)

    # Get Metrics for Linear Regression 
    elif type(model) == skl.linear_model.LinearRegression:
        metrics = pd.DataFrame.from_dict({"Measures": ["F-statistic", "Residual Std. Error", "Multiple R2", "Adj R2", "MSE", "RMSE",
                                                       "MAE", "Log Likelihood", "AIC", "BIC", "Residual Deviance"],
                                          "Values": [get_F_Statistic(model, x_train, y_train), get_Sigma(model, x_train, y_train), get_R2(model, x_train, y_train),
                                                     get_Adj_R2(model, x_train, y_train), get_MSE(model, x_train, y_train), get_RMSE(model, x_train, y_train),
                                                     get_MAE(model, x_train, y_train), get_log_lik(model, x_train, y_train), get_aic(model, x_train, y_train),
                                                     get_bic(model, x_train, y_train), get_residual_deviance(model, x_train, y_train)],
                                          "df1": [len(x_train.columns), len(x_train) - len(x_train.columns) - 1, "",
                                                 "", "", "", "",
                                                 "", "", "", len(x_train) - len(x_train.columns) - 1],
                                          "df2": [len(x_train) - len(x_train.columns) - 1, "", "",
                                                 "", "", "", "",
                                                 "", "", "", ""],
                                          "p.value": [get_F_Stat_P_Value(model, x_train, y_train), "", "", "",
                                                      "", "", "", "",
                                                      "", "", ""]})
        return metrics, get_pvalues(model, x_train, y_train)

    # Get Metrics for Decision Tree - Currently on hold 
    elif type(model) == skl.tree.DecisionTreeClassifier:
        print("A")



# Gets the AIC of the Model - Matches R Studio Values #
# Equal to 2*k - 2 * Log Liklihood #
def get_aic(model, x_train, y_train):

    # For Logistic Regression k = # of parameters + 1 
    if type(model) == skl.linear_model.LogisticRegression:
        return 2*(len(list(x_train.columns))+1) - 2 * get_log_lik(model, x_train, y_train)

    # For Linear regression k = # of parameters + 2 
    elif type(model) == skl.linear_model.LinearRegression:
        return 2*(len(list(x_train.columns))+2) - 2 * get_log_lik(model, x_train, y_train)



# Gets the AIC of the Model - Matches R Studio Values #
# Equal to k * log(n) - 2 * Log Liklihood #
# n is the number of observations #
def get_bic(model, x_train, y_train):

    # For Logistic Regression k = # of parameters + 1 
    if type(model) == skl.linear_model.LogisticRegression:
        return (len(list(x_train.columns))+1)*(np.log(len(x_train))) - 2 * get_log_lik(model, x_train, y_train)

    # For Linear regression k = # of parameters + 2 
    elif type(model) == skl.linear_model.LinearRegression:
        return (len(list(x_train.columns))+2)*(np.log(len(x_train))) - 2 * get_log_lik(model, x_train, y_train)



# Gets the Log Liklihood of a model #
def get_log_lik(model, x_train, y_train):
    # Below gets the probability from the rows represented by np.arange. 
    # It then gets the probability of the event happening, i.e. 0,1. 
    # It then sums the probability that those events happened to get the log loss 
    if type(model) == skl.linear_model.LogisticRegression:
        return model.predict_log_proba(x_train)[np.arange(len(y_train)),y_train].sum()

    # For Linear Regression, the log likelihood is equivalent to:
    # -(n/2) * log(2 * pi) - n * log(s) - 1/(s**2) * sum((y - pred_values)**2)
    # n is # of obs
    # s is the std dev of the difference of the predicted values and the actual values
    elif type(model) == skl.linear_model.LinearRegression:
        s = get_Sigma(model, x_train, y_train) * np.sqrt((len(y_train) - len(x_train.columns) - 1)/len(y_train))
        return -len(y_train)*np.log(2*np.pi)/2 - len(y_train)*np.log(s) - 1/(2*(s**2))*np.sum((y_train-model.predict(x_train))**2)



# Gets the null deviance, or the deviance of a model where the only predictor is the intercept. #
def get_null_deviance(model, x_train, y_train):
    x_dummy = np.full((len(x_train),len(list(x_train.columns))),0)

    # Returns the Log Likelihood of the null model without calling the function. For details on what's happening, look at the get_log_lik function 
    if type(model) == skl.linear_model.LogisticRegression:
        return -2*skl.linear_model.LogisticRegression(max_iter = 10000, penalty = None).fit(x_dummy,y_train).predict_log_proba(x_dummy)[np.arange(len(y_train)),y_train].sum()

    # Null Deviance isn't used for Linear Regression Model metrics 
    elif type(model) == skl.linear_model.LinearRegression:
        return print("Null Deviance doesn't make sense for Linear Regression")



# Gets the residual deviance of the model. For Linear Regression, this is know as SSE, or the Sum of Squares Error #
def get_residual_deviance(model, x_train, y_train):
    # For Logistical Regression, it is the sum of the log probabilities of the actual event occuring.
    # So, for the first row of records, if the actual outcome is 1, and the probability of that occuring is .8, you would add that to the total
    # If the second row is actually a 0, and the probability of that occuring is .2, you would add that to the total
    if type(model) == skl.linear_model.LogisticRegression:
        return -2*model.predict_log_proba(x_train)[np.arange(len(y_train)),y_train].sum()

    # For Linear Regression, it is just sum((actual - predicted)**2)
    elif type(model) == skl.linear_model.LinearRegression:
        return ((model.predict(x_train) - y_train)**2).sum()



# Gets the degrees of freedom for models. Which is just (n - p - 1), where n is the # of obs and p is # of parameters #
def get_residual_df(model, x_train, y_train):
    return len(x_train) - len(list(x_train.columns)) - 1



# Gets the degrees of freedom for the null model. Which is just (n - 1), where n is the # of obs. #
def get_null_df(model, x_train, y_train):
    return len(x_train) - 1



# Gets the traditiion R2 measure for Linear Regression #
def get_R2(model, x_train, y_train):
    return 1 - ((y_train - model.predict(x_train))**2).sum()/((y_train - np.mean(y_train))**2).sum()



# Gets the adjusted R2 measure for Linear Regression #
# Multiplies the calculated portion of the R2 measure by (n-1)/(n - p - 1) #
# n is the # of obs, p is the # of parameters #
def get_Adj_R2(model, x_train, y_train):
    return 1 - (((y_train - model.predict(x_train))**2).sum()/((y_train - np.mean(y_train))**2).sum())*((len(y_train)-1)/(len(y_train) - len(x_train.columns) - 1))



# Gets the McFadden R2 measure for Logistical Regression #
# Calculated by 1 - (resid. dev)/(null dev) #
def get_McFadden_R2(model, x_train, y_train):
    return 1-(get_residual_deviance(model,x_train,y_train)/get_null_deviance(model,x_train,y_train))



# Gets the Adjusted McFadden R2 for Logisitical Regression #
# Calculated by 1 -((resid. dev)/2 + p) / ((null dev)/2) #
# p is the number of parameters #
def get_McFadden_adj_R2(model, x_train, y_train):
    return 1-((get_residual_deviance(model,x_train,y_train)/2 + x_train.shape[1])/(get_null_deviance(model,x_train,y_train)/2))



# Gets the Efrons R2 Measure for Logistical Regression #
# Calculated by 1 - sum((y - (prob of 1 per row))**2) / sum((y - y_bar)**2) #
# y is the actual outcome, y_bar is the mean of the actual values #
# prob of 1 per row is the probability of the outcome being 1 based on the row. #
def get_Efrons_R2(model, x_train, y_train):
    return 1 - (np.sum(np.power(y_train - model.predict_proba(x_train)[:,1],2)) / np.sum(np.power(y_train - np.sum(y_train)/len(y_train),2)))



# Gets the Cox & Snell R2 Measure for Logistical Regression #
# Calculated by 1 - e^(- (null dev - resid. dev) / n), where n is the # of obs #
def get_Cox_Snell_R2(model, x_train, y_train):
    return 1 - np.exp(-(get_null_deviance(model, x_train, y_train) - get_residual_deviance(model, x_train, y_train))/len(y_train))



# Gets the Craig Uhler R2 Measure for Logistical Regression #
# Calculated by (1 - [e^(- (null dev)/2 + (resid. dev)/2)]**(2/n))) / (1 - [e^(- (null dev)/2)]**(2/n)) #
def get_Craig_Uhler_R2(model, x_train, y_train):
     return (1 - (np.exp(-get_null_deviance(model, x_train, y_train)/2 + get_residual_deviance(model, x_train, y_train)/2))**(2/len(x_train)))/(1 - (np.exp(-get_null_deviance(model, x_train, y_train)/2))**(2/len(x_train)))



# Gets the Accuracy of the model. So, how close did the model get to predicting all the values correctly. #
def get_Accuracy(model, x_train, y_train):
    y_hat = list(model.predict(x_train))
    correct = 0
    for i in range(len(y_hat)):
        if list(y_train)[i] == y_hat[i]:
            correct = correct + 1
    return(correct/len(y_train))



# Gets the Hosmer Lemeshow Test information. This tests how well the model's predicted probabilities #
# match the observed freequencies of events #
def get_Hosmer_Lemeshow_Test(model, x_train, y_train, num_groups = 10):
    Y_Yhat = pd.DataFrame({"y": y_train, "y_hat": model.predict_proba(x_train)[:,1]})
    Counts = pd.DataFrame(columns = ["Floor", "Ceiling", "y0","y1","yhat0","yhat1", "Total Count"])
    for i in range(num_groups):
        floor = np.percentile(Y_Yhat["y_hat"], 100 * (i/num_groups))
        if floor == np.min(Y_Yhat["y_hat"]):
            floor = floor - .000001
        ceiling = np.percentile(Y_Yhat["y_hat"], 100 * ((i+1)/num_groups))
        slice_Y_Yhat = Y_Yhat[(Y_Yhat["y_hat"] <= ceiling) & (Y_Yhat["y_hat"] > floor)]
        Counts = pd.concat([Counts if not Counts.empty else None, pd.DataFrame({"Floor": [floor], "Ceiling": [ceiling],
                                                  "y0": [len(slice_Y_Yhat) - np.sum(slice_Y_Yhat["y"])],
                                                  "y1": [np.sum(slice_Y_Yhat["y"])],
                                                  "yhat0": [len(slice_Y_Yhat) - np.sum(slice_Y_Yhat["y_hat"])],
                                                  "yhat1": [np.sum(slice_Y_Yhat["y_hat"])],
                                                  "Total Count": [len(slice_Y_Yhat)]})], ignore_index = True)
    Counts = Counts.assign(Chi_Squared = (Counts.y0 - Counts.yhat0)**2/Counts.yhat0 +
                           (Counts.y1 - Counts.yhat1)**2/Counts.yhat1)
    Chi_Squared = np.sum(Counts.Chi_Squared)
    return pd.DataFrame({"Chi_Squared": [np.sum(Counts.Chi_Squared)], "df": [num_groups - 2],
                        "p.value": [1 - scipy.stats.chi2.cdf(np.sum(Counts.Chi_Squared), num_groups - 2)]})


        
# Gets the P-Values for the individual parameters. This can be used to see how significant each parameter is. #
def get_pvalues(model, x_train, y_train):
    if type(model) == skl.linear_model.LogisticRegression:
        pred = model.predict_proba(x_train)
        coefs = np.concatenate([model.intercept_, model.coef_[0]])
        x_full = np.matrix(np.insert(np.array(x_train), 0, 1, axis = 1))
        ans = np.zeros((len(x_train.columns)+1, len(x_train.columns)+1))
        for i in range(len(y_train)):
            ans = ans + np.dot(np.transpose(x_full[i, :]), x_full[i, :]) * pred[i,1] * pred[i,0]
        se = np.sqrt(np.diag(np.linalg.inv(np.matrix(ans))))
        t = coefs/se
        return pd.DataFrame({"Terms" : ["Intercept"] + list(x_train.columns),
                             "Estimates" : coefs,
                             "Std. Error" : se,
                             "Statistics" : t,
                             "p.value" : (1 - scipy.stats.norm.cdf(abs(t))) * 2
                             })
    elif type(model) == skl.linear_model.LinearRegression:
        n = len(x_train)
        p = len(x_train.columns) + 1
        coef = np.append(model.intercept_, model.coef_)
        newX = pd.DataFrame({"Constant": np.ones(n)}).join(pd.DataFrame(x_train))
        MSE = np.sum((y_train - model.predict(x_train))**2)/(n - p)
        se_b = np.sqrt(MSE*(np.linalg.inv(np.dot(newX.T, newX)).diagonal()))
        return pd.DataFrame({
            "Parameter" : np.append("Intercept", x_train.columns),
            "Estimate" : coef,
            "Std. Error" : se_b,
            "t value" : coef/se_b,
            "p.value" : [2*(1 - scipy.stats.t.cdf(np.abs(i), n - p)) for i in coef/se_b]
            })



# Performs the Pearson Chi Square test on all parameters you are considering for the model. #
# Can be used to determine the signifcance of classification parameters for Logistical Regression models. #
# Will give you a value for continuous parameters, but should not be used. #
def Pearson_Chi_Square(x_train, y_train):
    Complete_Data = copy.deepcopy(x_train)
    Complete_Data[y_train.name] = y_train
    Total = len(x_train)
    Table = pd.DataFrame()
    for column in x_train.columns:
        Observed = pd.crosstab(Complete_Data[column], Complete_Data[y_train.name])
        Expected = copy.deepcopy(Observed).map(np.float64)
        for i in range(len(Observed)):
            for col in Observed.columns:
                Expected[col].iloc[i] = (np.sum(Observed.iloc[i])*np.sum(Observed[col]))/len(x_train)
        Table = pd.concat([Table if not Table.empty else None, pd.DataFrame({"Parameter": [column],
                                                                             "Chi_Squared Statistic": [np.sum(np.sum((Observed - Expected)**2/Expected,axis=0),axis=0)],
                                                                             "df": [(len(Observed)-1)*(len(Observed.columns)-1)],
                                                                             "p.value":[1 - scipy.stats.chi2.cdf(np.sum(np.sum((Observed - Expected)**2/Expected,axis=0),axis=0),(len(Observed)-1)*(len(Observed.columns)-1))]
                                                                             })], ignore_index = True)
    return Table



# Returns the confusion matrix for the previctions vs the actual results, so you can see where the most error predictions occur. #
def Confusion_Matrix(model, x_train, y_train):
    Data = pd.DataFrame({"Predicted": model.predict(x_train), "Actual": y_train})
    return pd.crosstab(Data["Predicted"], Data["Actual"])



# Can be used to get the predicted probability of a Logarithmic Function without using model.predict, but no need to use. #
def sigmoid_prediction_Function(model, x_train, y_train):
    return 1/(1+np.exp(-1*(np.dot(x_train, pd.DataFrame(model.coef_).T)+model.intercept_)))



# Gets the Residual Std. Error of a Linear Regression model. #
# Calculated by sqrt(SSE/(n - p - 1)), where n is the # of obs and p is the # of parameters #
def get_Sigma(model, x_train, y_train):
    return (((model.predict(x_train) - y_train)**2).sum()/(len(y_train)-len(x_train.columns)-1))**.5



# Gets the F_Statistic of the model to tell you if the model is sigificant over the null model.
# Calculated by (R2/p)/((1-R2)/(n - p - 1)), where R2 is the R Squared Value, n is the # of obs, and p is the # of parameters. #
def get_F_Statistic(model, x_train, y_train):
    R2 = get_R2(model, x_train, y_train)
    p = len(x_train.columns)
    return (R2/p)/((1-R2)/(len(y_train)-p-1))



# Based on the F Statistic, returns the P-Value, or level of significance for the model. #
def get_F_Stat_P_Value(model, x_train, y_train):
    return scipy.stats.f.sf(get_F_Statistic(model, x_train, y_train), len(x_train.columns), len(y_train) - len(x_train.columns) - 1)



# Calculates the MSE (Mean Square Error) #
# Calculated by finding the mean of ((actual - predicted)**2) #
def get_MSE(model, x_train, y_train):
    return np.mean((y_train - model.predict(x_train))**2)



# Calculates the RMSE (Root Mean Square Error) #
# Calculated by finding the sqrt(MSE)
def get_RMSE(model, x_train, y_train):
    return np.sqrt(np.mean((y_train - model.predict(x_train))**2))



# Calculates the MAE (Mean Absolute Error) #
# Calculated by finding the mean of abs(actual - predicted) #
def get_MAE(model, x_train, y_train):
    return np.mean(np.absolute(y_train - model.predict(x_train)))
