from sklearn.svm import LinearSVC

from configs.config import config

model = LinearSVC(C=config['model']['C'],
                  class_weight=config['model']['class_weight'],
                  max_iter=config['model']['max_iter'],
                  dual=False)
