import pandas as pd
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
import seaborn as sns
import theano.tensor as tt
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from scipy.stats import pearsonr


def sklog(X_train,Y_train, X_test, Y_test):

	clf = LogisticRegression()
	clf.fit(X_train, Y_train)

	p = clf.predict_proba(X_test)[:, 1]	

	return roc_auc_score(Y_test, p)


def predict(X, trace):
	
	um = trace['u'].mean(0)
	bm = trace['bs'].mean(0)
	p = um + np.dot(X, bm)

	return 1 / (1 + np.exp(-p))



if __name__ == '__main__':


	df = pd.read_csv('data.csv')

	df = df[df['native-country']==" United-States"]

	df['income'] = 1 * (df['income'] == " >50K")
	features = ['age', 'educ', 'hours']
	train = df[features + ['income']].sample(frac=0.7)
	test = df.drop(train.index)[features + ['income']]

	Y_train = np.array(train.income) 
	Y_test  = np.array(test.income) 
	X_test  = np.array(test[features])
	X_train = np.array(train[features])
	
	X_train = (X_train - X_train.mean(0)) / X_train.std(0)
	X_test  = (X_test - X_test.mean(0)) / X_test.std(0)


	with pm.Model() as logistic_model:
	    
		u = pm.Normal('u', 0, sd=10)
		fs = np.sign(np.asarray([pearsonr(x, Y_train)[0] for x in X_train.T]))
		b = pm.HalfNormal('b', sd=0.1, shape=X_train.shape[1])
		bs = pm.Deterministic('bs', tt.mul(fs, b))
		
		p = pm.math.invlogit(u + tt.dot(X_train, bs))

		likelihood = pm.Bernoulli('likelihood', p, observed=Y_train)

		tr = pm.sample(10000)

	ps = predict(X_test, tr)

	print('PyMC model acuracy {0:.3f}'.format(roc_auc_score(Y_test, ps)))
	print('Sklearn Model accuracy {0:.3f}'.format(sklog(X_train, Y_train, X_test, Y_test)))

