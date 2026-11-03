# Dataset: Clean/Dirty Road Classification

Source: https://www.kaggle.com/datasets/faizalkarim/cleandirty-road-classification (Kaggle, uploaded by Faizal Karim).
Check the licence on the Kaggle page before redistributing, and keep this credit.

* 231 unique images (111 clean, 120 dirty), resized to at most 800 px.
* `split.csv` lists every image with its label and split (train / val / test, 70 / 15 / 15, stratified, seed 42).
* Used to train the CityEye on-device street cleanliness model (`model/cleanliness_model.json`).
