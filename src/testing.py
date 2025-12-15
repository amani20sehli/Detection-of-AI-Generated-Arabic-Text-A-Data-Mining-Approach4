import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import joblib
from utils import elong_re, tokens_ar, stopwords, is_verb, is_dual

figures_dir = Path("../reports/figures/test")
figures_dir.mkdir(parents=True, exist_ok=True)
models_dir = Path("../reports/models")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def feat_elongations(text):
    count = 0
    for word in str(text).split():
        if elong_re.search(word):
            count += 1
    return count


def feat_periods(text):
    return str(text).count(".")


def feat_verbs(text):
    return len([t for t in tokens_ar(text) if is_verb(t)])


def feat_duals(text):
    return len([t for t in tokens_ar(text) if is_dual(t)])


def feat_entity_diversity(text):
    entities = [t for t in tokens_ar(text) if len(t) >= 3 and t not in stopwords and not is_verb(t)]
    if len(entities) < 10:
        return 0.0

    def calculate_mtld(tokens, threshold=0.72):
        factor = 0
        current = []
        for token in tokens:
            current.append(token)
            ttr = len(set(current)) / len(current)
            if ttr <= threshold:
                factor += 1
                current = []
        if current:
            factor += len(current) / len(tokens)
        return len(tokens) / factor if factor > 0 else len(tokens)

    mtld_score = (calculate_mtld(entities) + calculate_mtld(entities[::-1])) / 2
    return min(mtld_score / 100, 1.0)


print("Loading test data...")
test = pd.read_csv("../data/processed/test.csv")

print("Extracting features...")
X_test = np.array(
    [[feat_elongations(t), feat_periods(t), feat_verbs(t), feat_duals(t), feat_entity_diversity(t)] for t in
     test['text']])
y_test = test['label'].values

print("Loading models...")
scaler = joblib.load(models_dir / 'scaler.pkl')
lr = joblib.load(models_dir / 'lr.pkl')
svm = joblib.load(models_dir / 'svm.pkl')
rf = joblib.load(models_dir / 'rf.pkl')

X_test_scaled = scaler.transform(X_test)

test_results = []

y_pred_lr = lr.predict(X_test_scaled)
y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]
test_results.append({
    'model': 'Logistic Regression',
    'accuracy': accuracy_score(y_test, y_pred_lr),
    'precision': precision_score(y_test, y_pred_lr),
    'recall': recall_score(y_test, y_pred_lr),
    'f1': f1_score(y_test, y_pred_lr),
    'roc_auc': roc_auc_score(y_test, y_prob_lr)
})

y_pred_svm = svm.predict(X_test_scaled)
y_prob_svm = svm.predict_proba(X_test_scaled)[:, 1]
test_results.append({
    'model': 'SVM',
    'accuracy': accuracy_score(y_test, y_pred_svm),
    'precision': precision_score(y_test, y_pred_svm),
    'recall': recall_score(y_test, y_pred_svm),
    'f1': f1_score(y_test, y_pred_svm),
    'roc_auc': roc_auc_score(y_test, y_prob_svm)
})

y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]
test_results.append({
    'model': 'Random Forest',
    'accuracy': accuracy_score(y_test, y_pred_rf),
    'precision': precision_score(y_test, y_pred_rf),
    'recall': recall_score(y_test, y_pred_rf),
    'f1': f1_score(y_test, y_pred_rf),
    'roc_auc': roc_auc_score(y_test, y_prob_rf)
})

print("Loading BERT...")
tokenizer = AutoTokenizer.from_pretrained("asafaya/bert-base-arabic")
bert_model = AutoModel.from_pretrained("asafaya/bert-base-arabic").to(device).eval()


def extract_embeddings(texts, batch_size=32):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            embeddings.append(bert_model(**inputs).pooler_output.cpu().numpy())
    return np.vstack(embeddings)


X_test_bert = extract_embeddings(test['text'].tolist())


class FeedforwardNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.network(x)


model = FeedforwardNN().to(device)
model.load_state_dict(torch.load(models_dir / 'bert_nn.pt'))
model.eval()

with torch.no_grad():
    outputs = model(torch.FloatTensor(X_test_bert).to(device))
    y_prob_bert = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
    y_pred_bert = torch.argmax(outputs, dim=1).cpu().numpy()

test_results.append({
    'model': 'BERT + NN',
    'accuracy': accuracy_score(y_test, y_pred_bert),
    'precision': precision_score(y_test, y_pred_bert),
    'recall': recall_score(y_test, y_pred_bert),
    'f1': f1_score(y_test, y_pred_bert),
    'roc_auc': roc_auc_score(y_test, y_prob_bert)
})

results_df = pd.DataFrame(test_results).sort_values('f1', ascending=False)
results_df.to_csv("../reports/test_results.csv", index=False)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt='d', cmap='Greens', ax=axes[0, 0],
            xticklabels=['AI', 'Human'], yticklabels=['AI', 'Human'])
axes[0, 0].set_title('Logistic Regression')
sns.heatmap(confusion_matrix(y_test, y_pred_svm), annot=True, fmt='d', cmap='Purples', ax=axes[0, 1],
            xticklabels=['AI', 'Human'], yticklabels=['AI', 'Human'])
axes[0, 1].set_title('SVM')
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Oranges', ax=axes[1, 0],
            xticklabels=['AI', 'Human'], yticklabels=['AI', 'Human'])
axes[1, 0].set_title('Random Forest')
sns.heatmap(confusion_matrix(y_test, y_pred_bert), annot=True, fmt='d', cmap='Blues', ax=axes[1, 1],
            xticklabels=['AI', 'Human'], yticklabels=['AI', 'Human'])
axes[1, 1].set_title('BERT + NN')
plt.tight_layout()
plt.savefig(figures_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close()

test['prediction'] = y_pred_bert
test['probability'] = y_prob_bert
errors = test[test['label'] != test['prediction']]

if len(errors[errors['label'] == 0]) > 0:
    errors[errors['label'] == 0].nlargest(5, 'probability')[['text', 'probability']].to_csv(
        "../reports/false_positives.csv", index=False, encoding="utf-8-sig")

if len(errors[errors['label'] == 1]) > 0:
    errors[errors['label'] == 1].nsmallest(5, 'probability')[['text', 'probability']].to_csv(
        "../reports/false_negatives.csv", index=False, encoding="utf-8-sig")

pd.DataFrame({
    'category': ['FP', 'FN', 'Total'],
    'count': [len(errors[errors['label'] == 0]), len(errors[errors['label'] == 1]), len(errors)]
}).to_csv("../reports/error_summary.csv", index=False)

print("Done!")
