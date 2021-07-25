from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import mglearn
param_grid = {'n_estimators': [1,10,100,1000,10000],'max_features': [1,4,7,10,13]}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
results = pd.DataFrame(grid_search.cv_results_)
scores = np.array(results.mean_test_score).reshape(5, 5)
# 对交叉验证平均分数作图
mglearn.tools.heatmap(scores,xlabel='n_estimators',xticklabels=param_grid['n_estimators'],ylabel='max_features',yticklabels=param_grid['max_features'], cmap="viridis")
