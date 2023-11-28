import traceback
import warnings
import numpy as np
def _T_hdc_getTheModel(model, all2models = False):
    
    model_in = model
    try:
        new_model = np.array(model,dtype='<U10')
        model = np.array(model)
    except:
        raise ValueError("Model needs to be an array or list")

    if(model.ndim > 1):
        raise ValueError("The argument 'model' must be 1-dimensional")
    
    if type(model[0]) != np.str_:
        if np.any(np.apply_along_axis(np.isnan, 0, model)):
            raise ValueError("The argument 'model' cannot contain any Nan")

    ModelNames = np.array(["AKJBKQKDK", "AKBKQKDK", "ABKQKDK", "AKJBQKDK", "AKBQKDK", "ABQKDK", "AKJBKQKD", "AKBKQKD", "ABKQKD", "AKJBQKD", "AKBQKD", "ABQKD"])
    if type(model[0]) == np.str_:
        if model[0].isnumeric():
            model = model.astype(np.int_)
            #print(model)
        else:
            new_model = [np.char.upper(m) for m in model]

    if len(model) == 1 and new_model[0] == "ALL":
        if all2models:
            model = np.arange(0, 13)
        else:
            return "ALL"
        
    if type(model[0]) == np.int_:
        #print(model)
        qui = np.nonzero(np.isin(model, np.arange(0, 13)))[0]
        if len(qui) > 0:
            new_model[qui] = ModelNames[model[qui]]
        #print(new_model)
        
    qui = np.nonzero( np.invert(np.isin(new_model, ModelNames)))[0]
    if len(qui) > 0:
        if len(qui) == 1:
            msg = f'(e.g. {model_in[qui[0]]} is incorrect.)'

        else:
            msg = f'(e.g. {model_in[qui[0]]} or {model_in[qui[1]]} are incorrect.)'

        raise ValueError("Invalid model name " + msg)
    
    if np.max(np.unique(model, return_counts=True)[1]) > 1:
        warnings.warn("Values in 'model' argument should be unique.", UserWarning)

    mod_num = []
    for i in range(len(new_model)):
        mod_num.append(np.nonzero(new_model[i] == ModelNames)[0])
    mod_num = np.sort(np.unique(mod_num))
    new_model = ModelNames[mod_num]

    return new_model

a = ['ALL']
b = ['all']
c = ['all', 1]
d = ['all', 2]

tests = [a,b,c,d]

for model in tests:
    try:
        print(_T_hdc_getTheModel(model))
        #print("")
    except Exception as e:
        print("error " + str(e))
        #print(traceback.print_exc())
        