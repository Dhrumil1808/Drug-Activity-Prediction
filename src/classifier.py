import re
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn import cross_validation, linear_model
from sklearn.decomposition import TruncatedSVD
from imblearn.over_sampling import SMOTE

train_file = "train.dat"
test_file = "test.dat"

with open("train.dat", "r") as fr:
    lines = fr.readlines()

labels = [int(l[0]) for l in lines]
  
docs = [re.sub(r'[^\w]', ' ',l[1:]).split() for l in lines]

features = []

for doc in docs:
    line = [0]*100001
    for index, value in enumerate(doc):
        line[int(value)] = 1
    features.append(line)


truncatedsvd = TruncatedSVD(n_components=350)
truncatedsvd = truncatedsvd.fit(features)
reduced_features=np.array(truncatedsvd.transform(features))




sm = SMOTE(kind='svm',k_neighbors=9,m_neighbors=9)
features_resampled, labels_resampled = sm.fit_sample(reduced_features, labels)
#names=["Decision_tree","Ada_boost","Random_Forest","Gradient_Booster"]
names=["Decision_tree"]
#classifiers = [DecisionTreeClassifier(class_weight={0:0.8,1:2}),
#AdaBoostClassifier(n_estimators=10),RandomForestClassifier(n_estimators=58),GradientBoostingClassifier(loss='exponential',learning_rate=1,n_estimators=10)]

classifiers = [DecisionTreeClassifier(class_weight={0:0.8,1:2})]

with open("test.dat", "r") as fh:
    lines = fh.readlines()

test_labels = []
docs = [re.sub(r'[^\w]', ' ',l).split() for l in lines]

test_features = []

for doc in docs:
    line = [0]*100001
    for index, val in enumerate(doc):
        line[int(val)] = 1
    test_features.append(line)



test_reduced_features = np.array(truncatedsvd.transform(test_features))
for name, classifier in zip(names, classifiers):
    scores = cross_validation.cross_val_score(classifier, reduced_features, labels)
    classifier.fit(reduced_features, labels)
    test_predicted = classifier.predict(test_reduced_features)
    result_file = 'pr2_'+name+'_result'+'.dat'
    output = open(result_file, 'w')
    for t in test_predicted:
        output.write(str(t))
        output.write("\n")
    output.close()
    print "output written to pr2_Decision_Tree_result.dat"
