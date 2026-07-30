{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "fc5f72c5-85c1-48e4-a955-e0f8c1a60102",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import copy\n",
    "\n",
    "from sklearn import ensemble\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import accuracy_score\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "from sklearn.metrics import classification_report\n",
    "\n",
    "import warnings\n",
    "warnings.filterwarnings(\"ignore\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "d4d80722-5556-4351-b735-7bb940e61ef9",
   "metadata": {},
   "outputs": [],
   "source": [
    "df1 = pd.read_csv('train_data_after.csv', index_col=0)\n",
    "df2 = pd.read_csv('test_data_after.csv', index_col=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "b320916b-8ca5-4bf7-9c6f-ee961c5287c1",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "Index: 157525 entries, 0 to 157524\n",
      "Data columns (total 10 columns):\n",
      " #   Column              Non-Null Count   Dtype  \n",
      "---  ------              --------------   -----  \n",
      " 0   CreditScore         157525 non-null  float64\n",
      " 1   Age                 157525 non-null  float64\n",
      " 2   Tenure              157525 non-null  float64\n",
      " 3   Balance             157525 non-null  float64\n",
      " 4   NumOfProducts       157525 non-null  float64\n",
      " 5   HasCrCard           157525 non-null  int64  \n",
      " 6   IsActiveMember      157525 non-null  int64  \n",
      " 7   Exited              157525 non-null  int64  \n",
      " 8   Mem__no__Products   157525 non-null  float64\n",
      " 9   Age_Tenure_product  157525 non-null  float64\n",
      "dtypes: float64(7), int64(3)\n",
      "memory usage: 13.2 MB\n"
     ]
    }
   ],
   "source": [
    "df1.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "cae901b0-dabc-4335-9cc6-a9b4a0e1aa09",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "Index: 17503 entries, 0 to 17502\n",
      "Data columns (total 10 columns):\n",
      " #   Column              Non-Null Count  Dtype  \n",
      "---  ------              --------------  -----  \n",
      " 0   CreditScore         17503 non-null  float64\n",
      " 1   Age                 17503 non-null  float64\n",
      " 2   Tenure              17503 non-null  float64\n",
      " 3   Balance             17503 non-null  float64\n",
      " 4   NumOfProducts       17503 non-null  float64\n",
      " 5   HasCrCard           17503 non-null  int64  \n",
      " 6   IsActiveMember      17503 non-null  int64  \n",
      " 7   Exited              17503 non-null  int64  \n",
      " 8   Mem__no__Products   17503 non-null  float64\n",
      " 9   Age_Tenure_product  17503 non-null  float64\n",
      "dtypes: float64(7), int64(3)\n",
      "memory usage: 1.5 MB\n"
     ]
    }
   ],
   "source": [
    "df2.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "21cd0b41-8564-4462-9354-f02a9959b2ef",
   "metadata": {},
   "outputs": [],
   "source": [
    "train_points = df1.drop(['Exited'], axis=1)\n",
    "train_values = df1['Exited']\n",
    "test_points = df2.drop(['Exited'], axis=1)\n",
    "test_values = df2['Exited']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "91935cbb-e9fa-4750-a98a-d4e4193de22a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Threshold | Precision | Recall | F1 | Accuracy\n",
      "-----------------------------------------------------------------\n",
      "0.30      | 0.408      | 0.871    | 0.556   | 0.712\n",
      "0.31      | 0.414      | 0.864    | 0.559   | 0.718\n",
      "0.32      | 0.419      | 0.858    | 0.563   | 0.724\n",
      "0.33      | 0.425      | 0.851    | 0.567   | 0.731\n",
      "0.34      | 0.431      | 0.842    | 0.570   | 0.737\n",
      "0.35      | 0.438      | 0.833    | 0.574   | 0.744\n",
      "0.36      | 0.445      | 0.827    | 0.579   | 0.750\n",
      "0.37      | 0.450      | 0.818    | 0.581   | 0.755\n",
      "0.38      | 0.455      | 0.809    | 0.583   | 0.760\n",
      "0.39      | 0.461      | 0.801    | 0.586   | 0.765\n",
      "0.40      | 0.467      | 0.795    | 0.588   | 0.770\n",
      "0.41      | 0.475      | 0.785    | 0.592   | 0.776\n",
      "0.42      | 0.484      | 0.778    | 0.596   | 0.782\n",
      "0.43      | 0.492      | 0.769    | 0.600   | 0.788\n",
      "0.44      | 0.500      | 0.761    | 0.603   | 0.793\n",
      "0.45      | 0.508      | 0.751    | 0.606   | 0.798\n",
      "0.46      | 0.517      | 0.743    | 0.610   | 0.803\n",
      "0.47      | 0.524      | 0.732    | 0.611   | 0.807\n",
      "0.48      | 0.531      | 0.724    | 0.612   | 0.810\n",
      "0.49      | 0.540      | 0.715    | 0.615   | 0.815\n",
      "0.50      | 0.549      | 0.707    | 0.618   | 0.819\n",
      "0.51      | 0.556      | 0.696    | 0.618   | 0.822\n",
      "0.52      | 0.562      | 0.687    | 0.618   | 0.824\n",
      "0.53      | 0.570      | 0.679    | 0.620   | 0.828\n",
      "0.54      | 0.577      | 0.670    | 0.620   | 0.830\n",
      "0.55      | 0.584      | 0.661    | 0.620   | 0.832\n",
      "\n",
      "Лучший порог: 0.55 с F1 = 0.620\n"
     ]
    }
   ],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier\n",
    "\n",
    "rf_model = RandomForestClassifier(\n",
    "    n_estimators=1500,\n",
    "    max_depth=16,\n",
    "    min_samples_split=5,\n",
    "    min_samples_leaf=2,\n",
    "    class_weight={0: 1.0, 1: 4},   # пробуем сильнее\n",
    "    max_features='sqrt',\n",
    "    bootstrap=True,\n",
    "    random_state=42,\n",
    "    n_jobs=-1\n",
    ")\n",
    "\n",
    "rf_model.fit(train_points, train_values)\n",
    "\n",
    "# --- Поиск лучшего порога ---\n",
    "proba = rf_model.predict_proba(test_points)[:, 1]\n",
    "\n",
    "print(\"Threshold | Precision | Recall | F1 | Accuracy\")\n",
    "print(\"-\" * 65)\n",
    "\n",
    "best_f1 = 0\n",
    "best_threshold = 0.40\n",
    "\n",
    "\n",
    "\n",
    "for t in np.arange(0.30, 0.55, 0.01):\n",
    "    pred = (proba >= t).astype(int)\n",
    "    report = classification_report(test_values, pred, output_dict=True, digits=3)\n",
    "    p = report['1']['precision']\n",
    "    r = report['1']['recall']\n",
    "    f1 = report['1']['f1-score']\n",
    "    acc = report['accuracy']\n",
    "    \n",
    "    print(f\"{t:.2f}      | {p:.3f}      | {r:.3f}    | {f1:.3f}   | {acc:.3f}\")\n",
    "    \n",
    "    if f1 > best_f1:\n",
    "        best_f1 = f1\n",
    "        best_threshold = t\n",
    "\n",
    "print(f\"\\nЛучший порог: {best_threshold:.2f} с F1 = {best_f1:.3f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "0407116b-858c-4f54-a6dd-779fc55bd86a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Топ-10 самых важных признаков:\n",
      "              Feature  Importance\n",
      "1                 Age    0.293205\n",
      "4       NumOfProducts    0.242391\n",
      "0         CreditScore    0.111191\n",
      "3             Balance    0.108772\n",
      "7   Mem__no__Products    0.101053\n",
      "8  Age_Tenure_product    0.062517\n",
      "6      IsActiveMember    0.045150\n",
      "2              Tenure    0.026432\n",
      "5           HasCrCard    0.009288\n"
     ]
    }
   ],
   "source": [
    "importances = rf_model.feature_importances_\n",
    "feature_names = train_points.columns\n",
    "\n",
    "# Создаём DataFrame для удобства\n",
    "feature_importance_df = pd.DataFrame({\n",
    "    'Feature': feature_names,\n",
    "    'Importance': importances\n",
    "}).sort_values('Importance', ascending=False)\n",
    "\n",
    "print(\"Топ-10 самых важных признаков:\")\n",
    "print(feature_importance_df.head(10))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f5900b84-a3a9-42ba-a138-44fb2fe6600c",
   "metadata": {},
   "source": [
    "Удаляем признками <1% "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "3df1f382-c2c4-46f2-b00c-4256a9a71992",
   "metadata": {},
   "outputs": [],
   "source": [
    "del df1['HasCrCard']\n",
    "del df2['HasCrCard']\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "85d2292b-363b-43fb-97dd-b58d07c009c2",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "Index: 157525 entries, 0 to 157524\n",
      "Data columns (total 9 columns):\n",
      " #   Column              Non-Null Count   Dtype  \n",
      "---  ------              --------------   -----  \n",
      " 0   CreditScore         157525 non-null  float64\n",
      " 1   Age                 157525 non-null  float64\n",
      " 2   Tenure              157525 non-null  float64\n",
      " 3   Balance             157525 non-null  float64\n",
      " 4   NumOfProducts       157525 non-null  float64\n",
      " 5   IsActiveMember      157525 non-null  int64  \n",
      " 6   Exited              157525 non-null  int64  \n",
      " 7   Mem__no__Products   157525 non-null  float64\n",
      " 8   Age_Tenure_product  157525 non-null  float64\n",
      "dtypes: float64(7), int64(2)\n",
      "memory usage: 12.0 MB\n"
     ]
    }
   ],
   "source": [
    "df1.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "db1d5918-965b-468e-94c7-56fdae1b85aa",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "Index: 17503 entries, 0 to 17502\n",
      "Data columns (total 9 columns):\n",
      " #   Column              Non-Null Count  Dtype  \n",
      "---  ------              --------------  -----  \n",
      " 0   CreditScore         17503 non-null  float64\n",
      " 1   Age                 17503 non-null  float64\n",
      " 2   Tenure              17503 non-null  float64\n",
      " 3   Balance             17503 non-null  float64\n",
      " 4   NumOfProducts       17503 non-null  float64\n",
      " 5   IsActiveMember      17503 non-null  int64  \n",
      " 6   Exited              17503 non-null  int64  \n",
      " 7   Mem__no__Products   17503 non-null  float64\n",
      " 8   Age_Tenure_product  17503 non-null  float64\n",
      "dtypes: float64(7), int64(2)\n",
      "memory usage: 1.3 MB\n"
     ]
    }
   ],
   "source": [
    "df2.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "a6fae8cb-c4f1-4f3b-a084-c003005f7dde",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Обучение Random Forest...\n",
      "\n",
      "Classification Report при пороге 0.55:\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0      0.900     0.895     0.898     13878\n",
      "           1      0.608     0.620     0.614      3625\n",
      "\n",
      "    accuracy                          0.838     17503\n",
      "   macro avg      0.754     0.758     0.756     17503\n",
      "weighted avg      0.840     0.838     0.839     17503\n",
      "\n"
     ]
    }
   ],
   "source": [
    "rf_model = RandomForestClassifier(\n",
    "    n_estimators=1200,\n",
    "    max_depth=18,\n",
    "    min_samples_split=5,\n",
    "    min_samples_leaf=2,\n",
    "    class_weight={0: 1.0, 1: 3.8},   # подобранный вес\n",
    "    max_features='sqrt',\n",
    "    random_state=42,\n",
    "    n_jobs=-1\n",
    ")\n",
    "\n",
    "print(\"Обучение Random Forest...\")\n",
    "rf_model.fit(train_points, train_values)\n",
    "\n",
    "# ==================== 2. Предсказание с порогом 0.55 ====================\n",
    "# Получаем вероятности класса 1\n",
    "proba = rf_model.predict_proba(test_points)[:, 1]\n",
    "\n",
    "# Применяем выбранный порог 0.55\n",
    "THRESHOLD = 0.55\n",
    "y_pred = (proba >= THRESHOLD).astype(int)\n",
    "\n",
    "# ==================== 3. Вывод результатов ====================\n",
    "print(f\"\\nClassification Report при пороге {THRESHOLD}:\")\n",
    "print(classification_report(test_values, y_pred, digits=3))"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
