from sklearn import tree
from sklearn import ensemble

models = {
    'decision_tree_gini': tree.DecisionTreeClassifier(criterion='gini'),
    'decision_tree_enropy': tree.DecisionTreeClassifier(criterion='entropy'),
    'rf': ensemble.RandomForestClassifier()
}