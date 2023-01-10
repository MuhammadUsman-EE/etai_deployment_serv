#!pip install ktrain
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import ktrain
from ktrain import text
import numpy as np

master = 'https://raw.githubusercontent.com/mrreyesm/g6_tweets/main/master_dataset.parquet'
master = pd.read_parquet(master, engine='auto')
print(master.label.value_counts(), '\n')

X = np.asarray(master["clean_text"])
y = np.asarray(master["label"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

MODEL_NAME = 'distilbert-base-uncased'
t = text.Transformer(MODEL_NAME, maxlen=500, class_names=[0, 1])
trn = t.preprocess_train(X_train, y_train)
val = t.preprocess_test(X_test, y_test)
model = t.get_classifier()
learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=6)

learner.fit_onecycle(8e-5, 2)
#Get metrics
learner.validate(class_names=[1, 0])

learner.view_top_losses(n=1, preproc=t)

#predictor = ktrain.get_predictor(learner.model, preproc=t)

learner.save_model("/content/drive/MyDrive/distilbertmodelhf")