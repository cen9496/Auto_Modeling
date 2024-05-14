import pandas as pd
import sklearn as skl
import numpy as np
import copy
from Modeling.Metrics.Metrics import *

def AIC_Forward(model_to_use, x_train, y_train, x_test, y_test):
    model = copy.deepcopy(model_to_use)
    model_parameters = list(x_train.columns)
    use_parm = []
    for parameter in model_parameters:
        if parameter == model_parameters[0]:
            data_x = x_train.loc[:,[parameter]]
            fin_model = model.fit(data_x, y_train)
            aic = get_aic(fin_model, data_x, y_train)
            parm_used = parameter
        else:
            data_x = x_train.loc[:,[parameter]]
            step_model = model.fit(data_x, y_train)
            step_aic = get_aic(step_model, data_x, y_train)
            if step_aic < aic:
                fin_model = copy.deepcopy(step_model)
                parm_used = parameter
                aic = step_aic
    use_parm.append(parm_used)
    times = 1
    while True:
        update = False
        new_mod_parm = [parm for parm in model_parameters if parm not in use_parm]
        for parameter in new_mod_parm:
            data_x = x_train.loc[:,use_parm+[parameter]]
            step_model = model.fit(data_x, y_train)
            step_aic = get_aic(step_model, data_x, y_train)
            if step_aic < aic:
                fin_model = copy.deepcopy(step_model)
                parm_used = parameter
                update = True
                aic = step_aic
        if update == True:
            use_parm.append(parm_used)
        else:
            break
    return fin_model, use_parm

def AIC_Backward(model_to_use, x_train, y_train, x_test, y_test):
    model = copy.deepcopy(model_to_use)
    model_parameters = list(x_train.columns)
    use_parm = []
    for parameter in model_parameters:
        if parameter == model_parameters[0]:
            data_x = x_train.loc[:,[parm for parm in model_parameters if parm not in [parameter]]]
            fin_model = model.fit(data_x, y_train)
            aic = get_aic(fin_model, data_x, y_train)
            parm_used = parameter
        else:
            data_x = x_train.loc[:,[parm for parm in model_parameters if parm not in [parameter]]]
            step_model = model.fit(data_x, y_train)
            step_aic = get_aic(step_model, data_x, y_train)
            if step_aic < aic:
                fin_model = copy.deepcopy(step_model)
                parm_used = parameter
                aic = step_aic
    use_parm.append(parm_used)
    times = 1
    while True:
        update = False
        new_mod_parm = [parm for parm in model_parameters if parm not in use_parm]
        for parameter in new_mod_parm:
            data_x = x_train.loc[:,[parm for parm in model_parameters if parm not in use_parm+[parameter]]]
            step_model = model.fit(data_x, y_train)
            step_aic = get_aic(step_model, data_x, y_train)
            if step_aic < aic:
                fin_model = copy.deepcopy(step_model)
                parm_used = parameter
                update = True
                aic = step_aic
        if update == True:
            use_parm.append(parm_used)
        else:
            break
    return fin_model, [parm for parm in model_parameters if parm not in use_parm]

def AIC_Model(model, x_train, y_train, x_test, y_test, direction = "both"):
    model_parameters = list(x_train.columns)
    use_parm = []
    #Forwards
    if direction == "forward":
        return AIC_Forward(model, x_train, y_train, x_test, y_test)
    #Backwards
    elif direction == "backward":
        return AIC_Backward(model, x_train, y_train, x_test, y_test)
    # Both Directions
    elif direction == "both":
        mod_forw = copy.deepcopy(model)
        mod_back = copy.deepcopy(model)
        use_parm_forw = []
        use_parm_back = []
        for parameter in model_parameters:
            new_mod_parm_forw = [parm for parm in model_parameters if parm not in use_parm_forw]
            new_mod_parm_back = [parm for parm in model_parameters if parm not in use_parm_back]
            if parameter == model_parameters[0]:
                #Forwards
                data_x_forw = x_train.loc[:,[parameter]]
                model_forw = mod_forw.fit(data_x_forw, y_train)
                aic_forw = get_aic(model_forw, data_x_forw, y_train)
                parm_used_forw = parameter
                #Backwards
                data_x_back = x_train.loc[:,[parm for parm in model_parameters if parm not in [parameter]]]
                model_back = mod_back.fit(data_x_back, y_train)
                aic_back = get_aic(model_back, data_x_back, y_train)
                parm_used_back = parameter
            else:
                #Forwards
                data_x_forw = x_train.loc[:,[parameter]]
                step_model_forw = mod_forw.fit(data_x_forw, y_train)
                step_aic_forw = get_aic(step_model_forw, data_x_forw, y_train)
                if step_aic_forw < aic_forw:
                    model_forw = copy.deepcopy(step_model_forw)
                    parm_used_forw = parameter
                    aic_forw = step_aic_forw
                #Backwards
                data_x_back = x_train.loc[:,[parm for parm in model_parameters if parm not in [parameter]]]
                step_model_back = mod_back.fit(data_x_back, y_train)
                step_aic_back = get_aic(step_model_back, data_x_back, y_train)
                if step_aic_back < aic_back:
                    model_back = copy.deepcopy(step_model_back)
                    parm_used_back = parameter
                    aic_back = step_aic_back
        #Check Forward and Backward for Forward Model
        use_parm_forw = [parm_used_forw]
        while True:
            update = False
            new_mod_parm_forw = [parm for parm in model_parameters if parm not in use_parm_forw]
            for parameter in new_mod_parm_forw:
                data_x_forw = x_train.loc[:,[parameter]+use_parm_forw]
                step_model_forw = mod_forw.fit(data_x_forw, y_train)
                step_aic_forw = get_aic(step_model_forw, data_x_forw, y_train)
                if step_aic_forw < aic_forw:
                    model_forw = copy.deepcopy(step_model_forw)
                    parm_used_forw = parameter
                    update = True
                    aic_forw = step_aic_forw
            if update == True:
                use_parm_forw.append(parm_used_forw)
            for parameter in use_parm_forw:
                if len(use_parm_forw) <= 1:
                    break
                data_x_forw = x_train.loc[:,[parm for parm in use_parm_forw if parm not in [parameter]]]
                step_model_forw = mod_forw.fit(data_x_forw, y_train)
                step_aic_forw = get_aic(step_model_forw, data_x_forw, y_train)
                if step_aic_forw < aic_forw:
                    model_forw = copy.deepcopy(step_model_forw)
                    parm_used_forw = parameter
                    update = "Second Update"
                    aic_forw = step_aic_forw
            if update == "Second Update":
                use_parm_forw = [parm for parm in use_parm_forward if parm not in [parm_used_forw]]
            elif update == False:
                break
        #Check Forward and Backward for Backwards Model
        use_parm_back = [parm_used_back]
        while True:
            update = False
            new_mod_parm_back = [parm for parm in model_parameters if parm not in use_parm_back]
            for parameter in new_mod_parm_back:
                if len(new_mod_parm_back) <= 1:
                    break
                data_x_back = x_train.loc[:,[parm for parm in model_parameters if parm not in [parameter]+use_parm_back]]
                step_model_back = mod_back.fit(data_x_back, y_train)
                step_aic_back = get_aic(step_model_back, data_x_back, y_train)
                if step_aic_back < aic_back:
                    model_back = copy.deepcopy(step_model_back)
                    parm_used_back = parameter
                    update = True
                    aic_back = step_aic_back
            if update == True:
                use_parm_back.append(parm_used_back)
            for parameter in use_parm_back:
                if len(use_parm_back) <= 1:
                    break
                data_x_back = x_train.loc[:,[parm for parm in model_parameters if parm not in use_parm_back]+[parameter]]
                step_model_back = mod_back.fit(data_x_back, y_train)
                step_aic_back = get_aic(step_model_back, data_x_back, y_train)
                if step_aic_back < aic_back:
                    model_back = copy.deepcopy(step_model_back)
                    parm_used_back = parameter
                    update = "Second Update"
                    aic_back = step_aic_back
            if update == "Second Update":
                use_parm_back = [parm for parm in use_parm_back if parm not in [parameter]]
            elif update == False:
                break
        return model_forw, use_parm_forw, model_back, [parm for parm in model_parameters if parm not in use_parm_back]
    else:
        raise Exception("Please either set direction to forward, backward, or both.")
