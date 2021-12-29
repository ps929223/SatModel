
'''
결정트리(p226)
'''

def DT(train_input, test_input, train_target, test_target):
    from sklearn.tree import DecisionTreeClassifier
    dt=DecisionTreeClassifier(random_state=42)
    dt.fit(train_input, train_target)
    print(dt.score(train_input, train_target)) # 훈련세트
    print(dt.score(test_input, test_target)) # 테스트 세트
    return dt

def tree(dt, max_depth=1, filled=True):
    import matplotlib.pyplot as plt
    from sklearn.tree import plot_tree
    plt.figure(figsize=(10, 7))
    plot_tree(dt, max_depth=max_depth, filled=filled)


''' Test Code '''
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
# dt=DT(train_input, test_input, train_target, test_target)
# tree(dt)
#
# import matplotlib.pyplot as plt
# from sklearn.tree import plot_tree
#
# plt.figure(figsize=(10, 7))
# plot_tree(dt, max_depth=2, filled=True, feature_names=['alcohol','sugar','pH'])