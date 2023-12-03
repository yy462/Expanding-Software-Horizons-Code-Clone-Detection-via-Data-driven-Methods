import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
import os
from AST import createseparategraph

class CodeCloneDetector:
    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.model = RandomForestClassifier()  # 使用 RandomForestClassifier

    def load_data(self, file_path):
        data = pd.read_csv(file_path, sep='\t', header=None)
        data.columns = ['file1', 'file2', 'label']
        return data

    def extract_features(self, data, fit_vectorizer=False):
        createseparategraph(data,"",[],'')
        combined_code = []
        for _, row in data.iterrows():
            file1 = "./BCB" + row['file1'][1:]
            with open(file1, 'r') as file:
                file1_content = file.read()
            file2 = "./BCB" + row['file2'][1:]
            with open(file2, 'r') as file:
                file2_content = file.read()
            combined_code.append(file1_content + ' ' + file2_content)

        if fit_vectorizer:
            return self.vectorizer.fit_transform(combined_code)
        else:
            return self.vectorizer.transform(combined_code)

    def train(self, train_data):
        train_features = self.extract_features(train_data, fit_vectorizer=True)
        self.model.fit(train_features, train_data['label'])

    def predict(self, test_data):
        test_features = self.extract_features(test_data)
        return self.model.predict(test_features)

    def evaluate(self, test_data):
        predictions = self.predict(test_data)
        return classification_report(test_data['label'], predictions, output_dict=True)

    def save_predictions(self, predictions, file_name='random_forest_result_part.txt'):
        with open(file_name, 'w') as file:
            for pred in predictions:
                file.write(str(pred) + '\n')

detector = CodeCloneDetector()

train_data = detector.load_data('./BCB/traindata.txt')
test_data = detector.load_data('./BCB/testdata.txt')

detector.train(train_data)

predictions = detector.predict(test_data)

evaluation_metrics = detector.evaluate(test_data)
print(evaluation_metrics)

detector.save_predictions(predictions)
