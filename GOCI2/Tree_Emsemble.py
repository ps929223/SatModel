'''
트리앙상블 (p266)
'''

# 데이터 준비
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
wine = pd.read_csv('https://bit.ly/wine_csv_data')
data = wine[['alcohol','sugar','pH']].to_numpy()
target=wine['class'].to_numpy()
train_input, test_input, train_target, test_target=train_test_split(data, target, test_size=.2, random_state=42)

# 교차검증 수행
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=-1, random_state=42)
scores = cross_validate(rf, train_input, train_target, return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))

# 중요도 산출
rf.fit(train_input, train_target)
print(rf.feature_importances_)

# RF 자체평가 점수
rf = RandomForestClassifier(oob_score=True, n_jobs=-1, random_state=42) # OOB(out of bag): 부트스트랩 샘플에 포함되지 않고 남는 샘플
rf.fit(train_input, train_target)
print(rf.oob_score_)

# Extra Trees: 부트스트랩 샘플을 사용하지 않고 전체 훈련세트를 사용. 무작위 노드 분할
# 무작위분할로인해 성능은낮아지나, 많은 트리 앙상블 덕에 과대적합 막고 검증 세트 점수를 높이는 효과

from sklearn.ensemble import ExtraTreesClassifier
et = ExtraTreesClassifier(n_jobs=-1, random_state=42)
scores = cross_validate(et, train_input, train_target, return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))

# 중요도 산출
et.fit(train_input, train_target)
print(et.feature_importances_)

# 그레이디언트 부스팅: 경사하강법을 사용해 트리를 앙상블에 추가하는 형태

from sklearn.ensemble import GradientBoostingClassifier
gb=GradientBoostingClassifier(random_state=42)
scores = cross_validate(gb, train_input, train_target, return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
# 결과: 거의 과대적합되지 않음.


# 성능향상
gb=GradientBoostingClassifier(n_estimators=500, learning_rate=0.2, random_state=42) # 트리개수증가 500, 학습률증가 0.2
scores=cross_validate(gb, train_input, train_target, return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))

# GB의 중요도
gb.fit(train_input, train_target)
print(gb.feature_importances_)

# 히스토그램 기반 그레이디언트 부스팅
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
hgb=HistGradientBoostingClassifier(random_state=42)
scores = cross_validate(hgb, train_input, train_target, return_train_score=True)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))

# 중요도 산출
from sklearn.inspection import permutation_importance
hgb.fit(train_input, train_target)
result = permutation_importance(hgb, train_input, train_target, n_repeats=10, random_state=42, n_jobs=-1)
print(result.importances_mean) # trainset 반복하여 얻은 중요도, 평균, 표준편차

# testset 특성 중요도
result = permutation_importance(hgb, test_input, test_target, n_repeats=10, random_state=42, n_jobs=-1)
print(result.importances_mean)

# HistGradientBoostingClassifer 이용한 테스트세트에서 성능
hgb.score(test_input, test_target)

# XGBoost
from xgboost import XGBClassifier
xgb= XGBClassifier(tree_method='hist', random_state=42)
scores = cross_validate(xgb, train_input, train_target, return_train_score=True)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))

# LightGBM
from lightgbm import LGBMClassifier
lgb = LGBMClassifier(random_state=42)
scores = cross_validate(lgb, train_input, train_target, return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']),np.mean(scores['test_score']))