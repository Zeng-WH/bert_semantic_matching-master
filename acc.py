import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

test = pd.read_csv(r"/home1/CM2021/lym/QQbaseline/bert_semantic_matching-master/pre_label.csv")
pre_label = test['pre_label']
LCQMC = pd.read_csv(r"/home1/CM2021/lym/QQbaseline/bert_semantic_matching-master/data/lcqmc/LCQMC_test.csv")
test_label = LCQMC["label"]
test_accuracy = accuracy_score(test_label, pre_label)
print('test accuarcy:%.2f%%' % (test_accuracy * 100))
print("confusion matrix:")
results=confusion_matrix(test_label, pre_label)
print(results)