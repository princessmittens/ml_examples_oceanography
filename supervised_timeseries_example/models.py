from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree

def run_models(x_train, y_train, x_test, y_test):

   clf = RandomForestClassifier(n_estimators=10)
   clf.fit(x_train, y_train)
   score = clf.score(x_test, y_test)
   print("rf: " + str(score))


   nb = GaussianNB()
   nb.fit(x_train, y_train)
   nbaccuracy = nb.score(x_test, y_test)
   print("nb: " + str(nbaccuracy))


   dt = tree.DecisionTreeClassifier(random_state=10)
   dt.fit(x_train, y_train)
   dtaccuracy = dt.score(x_test, y_test)
   print("dt: " + str(dtaccuracy))

   clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
   clf.fit(x_train, y_train)
   svm_score = clf.score(x_test, y_test)
   print("svm " + str(score))
