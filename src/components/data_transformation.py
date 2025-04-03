import os
import sys
from src.exception import CustomException
from src.logger import logging

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from dataclasses import dataclass

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This function is responsible for Data Transformation
        """
        try:
            num_columns = ['age', 'bmi', 'children']
            cat_columns = ['sex', 'smoker', 'region']

            num_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder())
                ]
            )

            logging.info(f'Numerical Columns : {num_columns}')
            logging.info(f'Categorical Columns : {cat_columns}')

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, num_columns),
                    ('cat_pipeline', cat_pipeline, cat_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        This function will initiate the Data Transformation
        """           
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            logging.info('Reading train data and test data completed')

            logging.info('Obtaining Preprocessor Object')
            preprocessor_obj = self.get_data_transformer_object()

            target_column = 'charges'
            X_train = train_data.drop([target_column], axis=1)
            y_train = train_data[target_column]

            X_test = test_data.drop([target_column], axis=1)
            y_test = test_data[target_column]

            logging.info('Applying preprocessing object on train and test data')
            X_train_arr = preprocessor_obj.fit_transform(X_train)
            X_test_arr = preprocessor_obj.transform(X_test)

            train_arr = np.c_[X_train_arr, np.array(y_train)]
            test_arr = np.c_[X_test_arr, np.array(y_test)]

            logging.info('Saved Processing Object')
            # function in the utils.py
            save_object(
                file_path = self.transformation_config.preprocessor_obj_path,
                obj = preprocessor_obj
            )

            return(
                train_arr,
                test_arr,
                self.transformation_config.preprocessor_obj_path
            )

        except Exception as e:
            raise CustomException(e, sys)