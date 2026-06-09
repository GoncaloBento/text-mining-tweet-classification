# 4. Classical Machine Learning Classifiers
**Nova IMS — Text Mining 2025/2026**

This section documents our extensive experimentation with **traditional machine learning algorithms**. We systematically trained, hypertuned, and evaluated **16 total model configurations** representing six distinct algorithms: k-Nearest Neighbors (KNN), Logistic Regression, Multinomial Naive Bayes (MNB), Multi-Layer Perceptrons (MLP), Random Forests (RF), and XGBoost. All models are trained on our **Optimized TF-IDF unigrams+bigrams representation** ($N_{\text{features}}=25,000$, $\text{min\_df}=2$) and evaluated on our stratified validation fold.

---

## 📊 4.1. Comparative Classifier Leaderboard

The performance metrics of all 16 experiments logged on our rolling leaderboard (`outputs/results.csv`) are summarized below:

| Model ID | Classifier Family | Parameters & Search Spaces | Validation Accuracy | Macro Precision | Macro Recall | Macro F1-Score |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **LR-01** | **Logistic Regression** | **penalty=l2, C=1.0, balanced (CHAMPION)** | `0.7810` | `0.7041` | **`0.7201`** | **`0.7113`** |
| **NB-02** | **Multinomial NB** | alpha=0.1 (Lidstone Smoothing) | `0.7863` | `0.7534` | `0.6415` | `0.6793` |
| **MLP-02** | **Multi-Layer Perceptron** | hidden=(100, 50), lr=0.001 (Double Layer) | **`0.7873`** | `0.7366` | `0.6522` | `0.6830` |
| **LR-02** | **Logistic Regression** | penalty=l1, solver=saga, C=1.0, balanced | `0.7444` | `0.6601` | `0.6930` | `0.6733` |
| **MLP-01** | **Multi-Layer Perceptron** | hidden=(100,), lr=0.01 (Single Layer) | `0.7810` | `0.7443` | `0.6246` | `0.6621` |
| **RF-02** | **Random Forest** | n_estimators=200, max_depth=20, balanced | `0.7528` | `0.6790` | `0.6447` | `0.6589` |
| **RF-01** | **Random Forest** | n_estimators=100, max_depth=10, balanced | `0.7208` | `0.6379` | `0.6181` | `0.6244` |
| **NB-01** | **Multinomial NB** | alpha=1.0 (Laplace Smoothing) | `0.7339` | `0.7944` | `0.5000` | `0.5316` |
| **XGB-02** | **XGBoost** | max_depth=6, learning_rate=0.05, n_est=200 | `0.7376` | `0.7682` | `0.5247` | `0.5659` |
| **XGB-01** | **XGBoost** | max_depth=3, learning_rate=0.1, n_est=150 | `0.7349` | `0.7863` | `0.5088` | `0.5478` |
| **KNN-03** | **k-NN Baseline** | GridSearchCV Best (n_neighbors=3) | `0.6888` | `0.7305` | `0.4212` | `0.4247` |
| **KNN-01** | **k-NN Baseline** | n_neighbors=3 | `0.6888` | `0.7305` | `0.4212` | `0.4247` |
| **KNN-02** | **k-NN Baseline** | n_neighbors=7 | `0.6695` | `0.8276` | `0.3759` | `0.3463` |

---

## 📈 4.2. Scientific Insights & Discussion

Our empirical benchmarking yields four core scientific observations:

### 1. The Superiority of Logistic Regression L2 (`0.7113` Macro F1)
Although the Multi-Layer Perceptron achieves the highest overall accuracy (`0.7873`), **L2-regularized Logistic Regression is the overall performance champion**. In highly imbalanced text domains (where accuracy is biased by the 64.7% Neutral class), we must optimize for minority recall.
Logistic Regression, when equipped with `class_weight='balanced'`, adjusts decision thresholds mathematically. It scales the loss function inversely proportional to class frequencies, resulting in the highest recall on Bearish (`63.54%`) and Bullish (`67.79%`) tweets, yielding the best Macro F1 score.

### 2. Lidstone Smoothing Competitiveness in Naive Bayes (`0.6793` Macro F1)
Multinomial Naive Bayes performs exceptionally well when reducing the smoothing parameter ($\alpha=0.1$). Laplace smoothing ($\alpha=1.0$) overly dampens features in sparse high-dimensional n-gram vocabularies. Compressing the smoothing boundary via Lidstone smoothing ($\alpha=0.1$) allows the model to retain strong probabilistic cues for sentiment tokens while avoiding division-by-zero on unseen bigrams.

### 3. Non-Linear Accuracies in MLP (`0.7873` Accuracy)
The double-layer MLP Classifier (`hidden_layer_sizes=(100, 50)`) achieves the absolute highest validation accuracy. Its non-linear hidden layers and activation gates learn complex, multi-word contextual intersections (e.g. mapping the combination of target expectations and earnings to high-probability categories). However, its macro F1 lags slightly because, without explicit weight scaling, neural nets are highly prone to overfitting the majority class boundaries on small datasets.

### 4. Tree-based and Instance-based Failures
Ensemble trees (Random Forest, XGBoost) and k-Nearest Neighbors perform poorly:
* **KNN** is highly sensitive to the *Curse of Dimensionality*. In our $25,000$-dimensional TF-IDF space, Euclidean distances between sparse document vectors compress and become uniform, leading to near-random nearest neighbor boundaries.
* **Random Forest and XGBoost** struggle because decision trees make greedy orthogonal feature splits. High-dimensional, highly sparse text data consists of thousands of independent features with low individual predictive power, which dilutes split criteria and leads to shallow, uninformative trees.

---

## ⚙️ 4.3. Automated Tuning & Prediction Pipeline (`src/autotune.py`)

To streamline our submission workflow, we created a **Smart Dispatcher** retrainer script (`src/autotune.py`) to automatically orchestrate our predictions:

1. **Rolling Leaderboard Scan**: The tuner parses the leaderboard in `outputs/results.csv`, automatically locating the champion model based on the highest **Macro F1-Score**.
2. **Deep Learning Smart Dispatching**: If the champion model is a Hugging Face Transformer (e.g., DistilBERT, FinBERT), the script dynamically imports the dedicated trainer module and triggers its native PyTorch fine-tuning loop (`run_trainer(n_samples=None)`), safely handling complex tokenization without redundant code.
3. **Classical ML Refitting**: If the champion is a classical model (e.g., Logistic Regression), it refits the model configuration on **100% of the training data** ($N=9,543$) using our scikit-learn pipeline.
4. **Optimized Test Predictions**: Applies the appropriate predictor to the un-labeled test set (`data/test.csv`) and exports them to **`outputs/pred_best.csv`** in the exact formatted structure required for submission.
