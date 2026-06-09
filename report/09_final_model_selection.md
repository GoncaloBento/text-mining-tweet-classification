# 6. Final Model Selection
**Nova IMS — Text Mining 2025/2026**

Based on the empirical evidence from our extensive experiments, the final champion model selected for our official pipeline is the **Fine-Tuned DistilBERT** (`distilbert-base-uncased`).

## Justification:

1. **Absolute Performance Winner**: DistilBERT achieved the highest overall Macro F1-Score (**0.8143**) across all 30+ tested configurations, dramatically outperforming the best classical machine learning model (Logistic Regression, F1=0.7113).
2. **Contextual Semantic Superiority**: Our error analysis revealed that classical TF-IDF pipelines suffer from "Contextual Negation Blindness" and clause overriding because they ignore word order. DistilBERT's bi-directional self-attention layers natively resolve these syntactical dependencies (e.g., cleanly distinguishing *"fails to drop"* from *"drops"*), solving the primary failure mode of our baseline models.
3. **Imbalance Resilience**: DistilBERT attained strong recall on the difficult minority Bearish class (Class F1 = 0.734) simply through superior dense representation learning, entirely avoiding the need for artificial SMOTE oversampling or mathematical loss-scaling hacks (`class_weight='balanced'`).
4. **Efficiency vs. Domain Specificity**: Counter-intuitively, DistilBERT out-performed heavily domain-specific financial models like FinBERT (0.5911) and massive social media models like Twitter-RoBERTa (0.8011). DistilBERT proved to be the optimal "Goldilocks" model: it has enough general English representation to handle syntax, but is lightweight enough to achieve fast, highly stable convergence during fine-tuning on our compact dataset without catastrophically overfitting or collapsing (like DeBERTa-v3).

Consequently, `distilbert-base-uncased` has been fully integrated into our Agentic Dispatcher (`autotune.py`) and is deployed as the final model for `tm_final_xx.ipynb`.
