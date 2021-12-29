def StandardScaler(train_input, test_input):
    from sklearn.preprocessing import StandardScaler
    ss=StandardScaler()
    ss.fit(train_input)
    train_scaled=ss.transform(train_input)
    test_scaled=ss.transform(test_input)
    return train_scaled, test_scaled

def LR(train_scaled, test_scaled, train_target, test_target):
    from sklearn.linear_model import LogisticRegression
    lr=LogisticRegression()
    lr.fit(train_scaled, train_target)
    print(lr.score(train_scaled, train_target))
    print(lr.score(test_scaled, test_target))
    print(lr.coef_, lr.intercept_)
    return lr.coef_, lr.intercept_


# ''' Test Code '''
# import pandas as pd
# wine = pd.read_csv('https://bit.ly/wine_csv_data')
# wine.head()
# wine.info()
# wine.describe()
# data = wine[['alcohol','sugar','pH']].to_numpy()
# target=wine['class'].to_numpy()
#
# from sklearn.model_selection import train_test_split
# train_input, test_input, train_target, test_target=train_test_split(data, target, test_size=.2, random_state=42)
# print(train_input.shape, test_input.shape)
#
# train_scaled, test_scaled = StandardScaler(train_input, test_input)
#
# coef, intercept = LR(train_scaled, test_scaled, train_target, test_target)

