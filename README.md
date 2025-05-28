# Crop Harvest Prediction 🌽
A machine learning web-based application for predicting crop damage levels. Our application uses a trained model to assess the likelihood damage and classify it into appropriate categories.
## Features:
1. Predicts crop damage using real-world data
2. Uses XGBoost and Box-Cox transformation
3. Web interface with Flask
## Model:
The model type is an XGBoost classifier, and the data is preprocessed with one-hot encoding, feature scaling, and Box-Cox transformation
### Input:
  1. Estimated insects count
  2. Crop type
  3. Soil type
  4. Pesticide use category
  5. Number of pesticide doses per week
  6. Number of weeks used
  7. Number of weeks since quit
  8. Season
## Installation
1. Clone the repository
```
git clone https://github.com/yourusername/crop-harvest-prediction.git
cd crop-harvest-prediction
```
2. Install dependencies
```
pip install -r requirements.txt
```
3. Run the app
```
python app.py
```
The app will be open at `http://127.0.0.1:5000/`
## Credits
This project is made for Professor Ting's Introduction to Artificial Intelligence class at National Tsing Hua University, Taiwan
#### 🙍‍♂️Members:
1. 徐青霞 
2. 林佑銘
3. 黃少鋒
4. 莊誌強
5. 張嘉成
6. 周遠雄
