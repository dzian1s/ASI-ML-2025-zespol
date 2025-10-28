"""
This is a boilerplate pipeline 'data_preparation'
generated using Kedro 1.0.0
"""

from kedro.pipeline import Node, Pipeline, node, pipeline
from .nodes import *


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=load_raw,
                inputs=None,
                outputs='raw_data',
                name="load_raw_data"
        ),
            node(
                func=basic_clean,
                inputs='raw_data',
                outputs='clean_data',
                name="basic_clean"
            ),
            node(
                func=train_test_split,
                inputs='clean_data',
                outputs=['X_train', 'X_test', 'y_train', 'y_test'],
                name="train_test_split"
            ),
            node(
                func=train_baseline,
                inputs='train_test_split',
                outputs='model_baseline',
                name="train_baseline"
            ),
            node(
                func=evaluate,
                inputs='model_baseline',
                outputs='metrics_baseline',
                name="evaluate"
            )
        ]

    )
