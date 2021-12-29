
def Json2DF(path_json):
    import json
    import pandas as pd
    with open(path_json) as json_file:
        json_data = json.load(json_file)
    DF=pd.DataFrame(json_data)
    return DF

path_json= 'ais.json'
DF=Json2DF(path_json)