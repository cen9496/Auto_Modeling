import pandas as pd
import sklearn as skl
import numpy as np
import copy
import itertools
import random
from Modeling.Metrics.Metrics import *

# --------------------------------------------------------------------------------------------------------------------------- #
# Logarithmic Models #
# --------------------------------------------------------------------------------------------------------------------------- #

def Build_Log_Model(x_train, y_train, metric, iterations = 100000, print_steps = False): #you'd have to set it to True if you want to see what combination of parameters is better than the previously established parameters. 
    measures = ["Log Likelihood", "AIC", "BIC", "Residual Deviance", "McFadden R2",
               "McFadden Adjusted R2", "Efrons R2", "Cox Snell R2",
               "Craig Uhler R2", "Hosmer-Lemeshow", "Accuracy"]
    if metric not in measures:
        raise ValueError("Invalid metic type. Expect one of: %s" % measures)
    step = 0
    for combo in [list(x) for i in range(len(x_train.columns)) for x in itertools.combinations(x_train.columns, i+1)]:
        if combo == [list(x_train.columns)[0]]:
            use_model = skl.linear_model.LogisticRegression(max_iter = iterations, penalty=None).fit(x_train[combo], y_train)
            use_metrics, use_pvalues = get_metrics(use_model, x_train[combo], y_train)
            use_combo = combo
            if print_steps == True:
                step = step + 1
                print(f"-------  Step {step}  --------")
                print(use_metrics)
                print("")
                print(use_pvalues)
                print("")
                print(Confusion_Matrix(use_model, x_train[combo], y_train))
                print("")
        else:
            iter_model = skl.linear_model.LogisticRegression(max_iter = iterations, penalty=None).fit(x_train[combo], y_train)
            iter_metrics, iter_pvalues = get_metrics(iter_model, x_train[combo], y_train)
            iter_combo = combo
            if metric in ["AIC", "BIC", "Residual Deviance"]:
                if iter_metrics[iter_metrics["Measures"] == metric]["Values"].values[0] < use_metrics[use_metrics["Measures"] == metric]["Values"].values[0]:
                    use_model = copy.deepcopy(iter_model)
                    use_metrics, use_pvalues = copy.deepcopy(iter_metrics), copy.deepcopy(iter_pvalues)
                    use_combo = copy.deepcopy(iter_combo)
                    if print_steps == True:
                        step = step + 1
                        print(f"-------  Step {step}  --------")
                        print(use_metrics)
                        print("")
                        print(use_pvalues)
                        print("")
                        print(Confusion_Matrix(use_model, x_train[combo], y_train))
                        print("")
            else:
                if iter_metrics[iter_metrics["Measures"] == metric]["Values"].values[0] > use_metrics[use_metrics["Measures"] == metric]["Values"].values[0]:
                    use_model = copy.deepcopy(iter_model)
                    use_metrics, use_pvalues = copy.deepcopy(iter_metrics), copy.deepcopy(iter_pvalues)
                    use_combo = copy.deepcopy(iter_combo)
                    if print_steps == True:
                        step = step + 1
                        print(f"-------  Step {step}  --------")
                        print(use_metrics)
                        print("")
                        print(use_pvalues)
                        print("")
                        print(Confusion_Matrix(use_model, x_train[combo], y_train))
                        print("")
    return use_model, use_metrics, use_pvalues, use_combo

# This will break your training data into "buckets" where the percentages are the how you want to break the set up
# For example, if you want to have models for 20% and 30% of your data being the test set, you would use [0.2, 0.3]
# If you only want to see for 20% of your data being test data, you'd set it to [0.2]
def Build_Log_Model_Buckets(x_train, y_train, metric, test_percentages = [.2,.3,.4], iterations = 10000, print_steps = False):
    seed = random.randrange(0,1000,1)
    models = {}
    measures = ["Log Likelihood", "AIC", "BIC", "Residual Deviance", "McFadden R2",
               "McFadden Adjusted R2", "Efrons R2", "Cox Snell R2",
               "Craig Uhler R2", "Hosmer-Lemeshow", "Accuracy"]
    if metric not in measures:
        raise ValueError("Invalid metic type. Expect one of: %s" % measures)
    for i in range(len(test_percentages)):
        print("This is the best model where ", test_percentages[i]*100,"% of the data is in the test data for random state ", seed)
        input_train, input_test, output_train, output_test = skl.model_selection.train_test_split(x_train, y_train, test_size = test_percentages[i], random_state = seed)
        model, metrics, p_values, use_combo = Build_Log_Model(input_train, output_train, metric, iterations, print_steps)
        models[test_percentages[i]] = {"Model": copy.deepcopy(model), "Metrics" : copy.deepcopy(metrics), "p.values" : copy.deepcopy(p_values), "Parameters" : copy.deepcopy(use_combo)}
        print("")
        print("The model's accuracy at predicting the test data is: ", get_Accuracy(model, input_test[use_combo], output_test))
        print("")
        print("The model's confusion matrix is: ")
        print(Confusion_Matrix(model, input_test[use_combo], output_test))
    return(models)


# --------------------------------------------------------------------------------------------------------------------------- #
# Linear Models #
# --------------------------------------------------------------------------------------------------------------------------- #

def Build_Linear_Model(x_train, y_train, metric, iterations = 100000, print_steps = False): #you'd have to set it to True if you want to see what combination of parameters is better than the previously established parameters. 
    measures = ["Log Likelihood", "AIC", "BIC", "Residual Deviance", "Multiple R2", "Adj R2",
                "MSE", "RMSE", "MAE"]
    if metric not in measures:
        raise ValueError("Invalid metic type. Expect one of: %s" % measures)
    step = 0
    for combo in [list(x) for i in range(len(x_train.columns)) for x in itertools.combinations(x_train.columns, i+1)]:
        if combo == [list(x_train.columns)[0]]:
            use_model = skl.linear_model.LinearRegression(n_jobs = iterations).fit(x_train[combo], y_train)
            use_metrics, use_pvalues = get_metrics(use_model, x_train[combo], y_train)
            use_combo = combo
            if print_steps == True:
                step = step + 1
                print(f"-------  Step {step}  --------")
                print(use_metrics)
                print("")
                print(use_pvalues)
                print("")
        else:
            iter_model = skl.linear_model.LinearRegression(n_jobs = iterations).fit(x_train[combo], y_train)
            iter_metrics, iter_pvalues = get_metrics(iter_model, x_train[combo], y_train)
            iter_combo = combo
            if metric in ["AIC", "BIC", "Residual Deviance", "MSE", "RMSE", "MAE"]:
                if iter_metrics[iter_metrics["Measures"] == metric]["Values"].values[0] < use_metrics[use_metrics["Measures"] == metric]["Values"].values[0]:
                    use_model = copy.deepcopy(iter_model)
                    use_metrics, use_pvalues = copy.deepcopy(iter_metrics), copy.deepcopy(iter_pvalues)
                    use_combo = copy.deepcopy(iter_combo)
                    if print_steps == True:
                        step = step + 1
                        print(f"-------  Step {step}  --------")
                        print(use_metrics)
                        print("")
                        print(use_pvalues)
                        print("")
            else:
                if iter_metrics[iter_metrics["Measures"] == metric]["Values"].values[0] > use_metrics[use_metrics["Measures"] == metric]["Values"].values[0]:
                    use_model = copy.deepcopy(iter_model)
                    use_metrics, use_pvalues = copy.deepcopy(iter_metrics), copy.deepcopy(iter_pvalues)
                    use_combo = copy.deepcopy(iter_combo)
                    if print_steps == True:
                        step = step + 1
                        print(f"-------  Step {step}  --------")
                        print(use_metrics)
                        print("")
                        print(use_pvalues)
                        print("")
    return use_model, use_metrics, use_pvalues, use_combo

# This will break your training data into "buckets" where the percentages are the how you want to break the set up
# For example, if you want to have models for 20% and 30% of your data being the test set, you would use [0.2, 0.3]
# If you only want to see for 20% of your data being test data, you'd set it to [0.2]
def Build_Linear_Model_Buckets(x_train, y_train, metric, test_percentages = [.2,.3,.4], iterations = 10000, print_steps = False):
    seed = random.randrange(0,1000,1)
    models = {}
    measures = ["Log Likelihood", "AIC", "BIC", "Residual Deviance", "McFadden R2",
               "McFadden Adjusted R2", "Efrons R2", "Cox Snell R2",
               "Craig Uhler R2", "Hosmer-Lemeshow", "Accuracy"]
    if metric not in measures:
        raise ValueError("Invalid metic type. Expect one of: %s" % measures)
    for i in range(len(test_percentages)):
        print("This is the best model where ", test_percentages[i]*100,"% of the data is in the test data for random state ", seed)
        input_train, input_test, output_train, output_test = skl.model_selection.train_test_split(x_train, y_train, test_size = test_percentages[i], random_state = seed)
        model, metrics, p_values, use_combo = Build_Linear_Model(input_train, output_train, metric, iterations, print_steps)
        models[test_percentages[i]] = {"Model": copy.deepcopy(model), "Metrics" : copy.deepcopy(metrics), "p.values" : copy.deepcopy(p_values), "Parameters" : copy.deepcopy(use_combo)}
        print("")
        print("Below are the metrics of the test data using the model: ")
        print("")
        t_metrics, t_pval = get_metrics(model, input_test[use_combo], output_test)
        print(t_metrics)
        print("")
    return(models)
