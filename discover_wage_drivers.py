"""
title: Discover Wage Drivers
description: Key operational workflow, which fits a NN model for wages and discover wages drivers
"""

#
# 0.
#
import os
import datetime

APP_HOME = "/content/drive/MyDrive/scapping_lab_market"
os.chdir(APP_HOME)

from spark_session_management import create_spark_session
from ml_samples import get_ml_samples, get_training_metadata
from deep_learning import train_wages_model
from wages_drivers import mine_marginal_effects_with_proportion, plot_effects_and_proportions, plot_effects_rank
from commit_inference import commit_inference
from datetime import datetime

#
# 1.
#
spark = create_spark_session()
train_X, train_y, val_X, val_y = get_ml_samples(spark, test_size=0.5)

full_sample_size=get_training_metadata()['full_sample_size']
feature_names=get_training_metadata()['feature_names']

#
# 2.
#
wages_model=train_wages_model(train_X, train_y, val_X, val_y, num_epochs=150,  threshold=.005, num_conepts=12, window_size=15)

#
# 3.
#
marginal_effects = mine_marginal_effects_with_proportion(wages_model, feature_names, val_X )
marginal_effects


result = commit_inference(
    segment_keyword=get_training_metadata()['segment_keyword'],
    start_date=datetime(2024, 1, 15),
    end_date=datetime(2025, 12, 15),
    sample_size=get_training_metadata()['full_sample_size'],
    vocabulary_size=len(get_training_metadata()['feature_names']),
    keywords=marginal_effects
)
#
print(result)
