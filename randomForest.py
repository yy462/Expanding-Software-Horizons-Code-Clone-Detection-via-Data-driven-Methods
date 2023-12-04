import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
import os
from AST import createseparategraph

class CodeCloneDetector:
    def __init__(self):
        self.vectorizer = CountVectorizer()  # Initialize a CountVectorizer for text feature extraction
        self.model = RandomForestClassifier()  # Initialize the RandomForest Classifier model

    def load_data(self, file_path):
        """
        Loads data from a given file path into a Pandas DataFrame.
        """
        data = pd.read_csv(file_path, sep='\t', header=None)
        data.columns = ['file1', 'file2', 'label']  # Naming the columns
        return data

    def extract_features(self, data, fit_vectorizer=False):
        """
        Extracts features from the code by reading file contents and vectorizing them.
        """
        createseparategraph(data,"",[],'')  # Use AST to create a separate graph representation (not fully implemented here)
        combined_code = []
        for _, row in data.iterrows():
            # Read contents of the two code files and combine them
            file1 = "./BCB" + row['file1'][1:]
            with open(file1, 'r') as file:
                file1_content = file.read()
            file2 = "./BCB" + row['file2'][1:]
            with open(file2, 'r') as file:
                file2_content = file.read()
            combined_code.append(file1_content + ' ' + file2_content)

        # Vectorize the combined code
        if fit_vectorizer:
            return self.vectorizer.fit_transform(combined_code)
        else:
            return self.vectorizer.transform(combined_code)

    def train(self, train_data):
        """
        Trains the RandomForest model using the provided training data.
        """
        train_features = self.extract_features(train_data, fit_vectorizer=True)
        self.model.fit(train_features, train_data['label'])

    def predict(self, test_data):
        """
        Predicts labels for the test data using the trained model.
        """
        test_features = self.extract_features(test_data)
        return self.model.predict(test_features)

    def evaluate(self, test_data):
        """
        Evaluates the model's performance on the test data.
        """
        predictions = self.predict(test_data)
        return classification_report(test_data['label'], predictions, output_dict=True)

    def save_predictions(self, predictions, file_name='random_forest_result_part.txt'):
        """
        Saves the model's predictions to a file.
        """
        with open(file_name, 'w') as file:
            for pred in predictions:
                file.write(str(pred) + '\n')

# Example usage of the CodeCloneDetector class
detector = CodeCloneDetector()

# Loading training and testing data
train_data = detector.load_data('./BCB/traindata.txt')
test_data = detector.load_data('./BCB/testdata.txt')

# Training the model
detector.train(train_data)

# Making predictions on the test data
predictions = detector.predict(test_data)

# Evaluating the model
evaluation_metrics = detector.evaluate(test_data)
print(evaluation_metrics)

# Saving the predictions
detector.save_predictions(predictions)
