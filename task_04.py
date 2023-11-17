from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import StackingClassifier
from imutils import paths
import cv2

CAT = 1
DOG = 0

def extract_histogram(image, bins=(8, 8, 8)):
  hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
  cv2.normalize(hist, hist)
  return hist.flatten()

def readImgData(img_paths):
  X = [] #данные гистограммы картинки
  Y = [] #класс объектв 0 или 1

  for path in img_paths:
    image = cv2.imread(path, 1)
    is_cat = CAT if 'cat' in path else DOG
    histogram = extract_histogram(image)
    X.append(histogram) 
    Y.append(is_cat)

  return [X, Y]

#сборка путей к картинкам
img_dir = "./task_04_train"
img_paths = sorted(list(paths.list_images(img_dir)))

[X, Y] = readImgData(img_paths)

svm = LinearSVC(C=1.25, random_state=80)

bagging = BaggingClassifier(
  base_estimator= DecisionTreeClassifier(
    criterion='entropy',
    min_samples_leaf=10,
    max_leaf_nodes=20,
    random_state=80
  ),
  n_estimators=13,
  random_state=80
)

forest = RandomForestClassifier(n_estimators=13, criterion='entropy', min_samples_leaf=10, max_leaf_nodes=20, random_state=80)

#Логистическая регрессия 
logistic = LogisticRegression(solver='lbfgs', random_state=80)

base_estimators = [('SVM', svm), ('Bagging DT', bagging), ('DesicionForest', forest)]
sclf = StackingClassifier(estimators=base_estimators, final_estimator=logistic, cv=2)

sclf.fit(X, Y)
score = sclf.score(X,Y)

print(f'Доля правильной классификации (Accuracy): {score}')

#Предсказание изображение
imgs_to_predict = ['dog.1049.jpg', 'dog.1028.jpg', 'dog.1011.jpg', 'cat.1016.jpg']
paths = []

test_img_dir = './task_04_test'
for img_name in imgs_to_predict:
  paths.append(test_img_dir + '/' + img_name)

[X_pred, Y_pred] = readImgData(paths)

class_probabilities = sclf.predict_proba(X_pred)
class_probabilities = [prob[1] for prob in class_probabilities]
print(f'Вроятность отнесения изображений к классу 1: {class_probabilities}')