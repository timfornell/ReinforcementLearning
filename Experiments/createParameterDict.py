def createParameterDict(REQUIRED_PARAMS_FROM_ENVIRONMENT):
    param_dict = dict()
    for param in REQUIRED_PARAMS_FROM_ENVIRONMENT:
        param_dict[param] = ""

    return param_dict
