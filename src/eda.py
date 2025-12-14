import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
from utils import normalize_ar, tokens_ar, remove_stopwords, stem_tokens, stopwords

plt.rcParams['font.family'] = 'Arial'
sns.set_style("whitegrid")

def preprocess_pipeline(text):
    normalized = normalize_ar(text)
    tokens = tokens_ar(text)
    tokens = [t for t in tokens if len(t) >= 3]
    tokens_no_stop = remove_stopwords(tokens)
    stemmed = stem_tokens(tokens_no_stop)
    return {
        'normalized': normalized,
        'tokens': tokens,
        'tokens_no_stop': tokens_no_stop,
        'stemmed': stemmed
    }

def get_ngrams(tokens, n):
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

print("Loading data...")
train = pd.read_csv("../data/processed/train.csv")
val = pd.read_csv("../data/processed/val.csv")
df = pd.concat([train, val], ignore_index=True)
print(f"Total: {len(df)} samples")

print("\nPreprocessing...")
df['processed'] = df['text'].apply(preprocess_pipeline)
df['tokens'] = df['processed'].apply(lambda x: x['tokens'])
df['tokens_no_stop'] = df['processed'].apply(lambda x: x['tokens_no_stop'])

df['word_count'] = df['tokens'].apply(len)
df['word_count_no_stop'] = df['tokens_no_stop'].apply(len)
df['avg_word_length'] = df['tokens'].apply(lambda x: np.mean([len(w) for w in x]) if x else 0)
df['sentence_count'] = df['text'].apply(lambda x: x.count('.') + x.count('!') + x.count('?'))
df['ttr'] = df.apply(lambda x: len(set(x['tokens'])) / len(x['tokens']) if x['tokens'] else 0, axis=1)
df['punct_count'] = df['text'].apply(lambda x: sum(1 for c in str(x) if c in '.,;:!?'))
df['stopword_ratio'] = df.apply(lambda x: 1 - (len(x['tokens_no_stop']) / len(x['tokens'])) if x['tokens'] else 0, axis=1)

human = df[df['label'] == 1]
ai = df[df['label'] == 0]

print("\n" + "="*10)
print("STATISTICAL ANALYSIS")
print("="*10)

metrics = ['word_count', 'word_count_no_stop', 'avg_word_length', 'sentence_count', 'ttr', 'punct_count', 'stopword_ratio']
stats = pd.DataFrame({
    'metric': metrics,
    'human_mean': [human[m].mean() for m in metrics],
    'human_std': [human[m].std() for m in metrics],
    'ai_mean': [ai[m].mean() for m in metrics],
    'ai_std': [ai[m].std() for m in metrics]
})

stats.to_csv("../reports/eda_statistics.csv", index=False, encoding="utf-8-sig")
print("\n" + stats.to_string(index=False))

print("\n" + "="*20)
print("VISUALIZATION")


fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Distribution Comparison: Human vs AI', fontsize=16)

for idx, metric in enumerate(['word_count', 'avg_word_length', 'sentence_count', 'ttr', 'punct_count', 'stopword_ratio']):
    ax = axes[idx // 3, idx % 3]
    ax.hist(human[metric], bins=30, alpha=0.5, label='Human', color='blue')
    ax.hist(ai[metric], bins=30, alpha=0.5, label='AI', color='red')
    ax.set_xlabel(metric.replace('_', ' ').title())
    ax.set_ylabel('Frequency')
    ax.legend()

plt.tight_layout()
plt.savefig("../reports/distributions.png", dpi=300, bbox_inches='tight')
print("Saved: distributions.png")

print("\nGenerating word clouds...")
all_human_tokens = ' '.join([' '.join(tokens) for tokens in human['tokens_no_stop']])
all_ai_tokens = ' '.join([' '.join(tokens) for tokens in ai['tokens_no_stop']])

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

wc_human = WordCloud(width=800, height=400, background_color='white',
                     font_path='arial.ttf', colormap='Blues').generate(all_human_tokens)
axes[0].imshow(wc_human, interpolation='bilinear')
axes[0].set_title('Human-Written Text', fontsize=16)
axes[0].axis('off')

wc_ai = WordCloud(width=800, height=400, background_color='white',
                  font_path='arial.ttf', colormap='Reds').generate(all_ai_tokens)
axes[1].imshow(wc_ai, interpolation='bilinear')
axes[1].set_title('AI-Generated Text', fontsize=16)
axes[1].axis('off')

plt.tight_layout()
plt.savefig("../reports/wordclouds.png", dpi=300, bbox_inches='tight')
print("Saved: wordclouds.png")

print("\n" + "="*10)
print("N-GRAM ANALYSIS")


print("\nExtracting n-grams...")
all_bigrams_human = []
all_trigrams_human = []
for tokens in human['tokens_no_stop']:
    all_bigrams_human.extend(get_ngrams(tokens, 2))
    all_trigrams_human.extend(get_ngrams(tokens, 3))

all_bigrams_ai = []
all_trigrams_ai = []
for tokens in ai['tokens_no_stop']:
    all_bigrams_ai.extend(get_ngrams(tokens, 2))
    all_trigrams_ai.extend(get_ngrams(tokens, 3))

top_bigrams_human = Counter(all_bigrams_human).most_common(20)
top_bigrams_ai = Counter(all_bigrams_ai).most_common(20)
top_trigrams_human = Counter(all_trigrams_human).most_common(20)
top_trigrams_ai = Counter(all_trigrams_ai).most_common(20)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

axes[0, 0].barh([x[0] for x in top_bigrams_human][::-1], [x[1] for x in top_bigrams_human][::-1], color='blue')
axes[0, 0].set_title('Top 20 Bigrams - Human', fontsize=14)
axes[0, 0].set_xlabel('Frequency')

axes[0, 1].barh([x[0] for x in top_bigrams_ai][::-1], [x[1] for x in top_bigrams_ai][::-1], color='red')
axes[0, 1].set_title('Top 20 Bigrams - AI', fontsize=14)
axes[0, 1].set_xlabel('Frequency')

axes[1, 0].barh([x[0] for x in top_trigrams_human][::-1], [x[1] for x in top_trigrams_human][::-1], color='blue')
axes[1, 0].set_title('Top 20 Trigrams - Human', fontsize=14)
axes[1, 0].set_xlabel('Frequency')

axes[1, 1].barh([x[0] for x in top_trigrams_ai][::-1], [x[1] for x in top_trigrams_ai][::-1], color='red')
axes[1, 1].set_title('Top 20 Trigrams - AI', fontsize=14)
axes[1, 1].set_xlabel('Frequency')

plt.tight_layout()
plt.savefig("../reports/ngrams.png", dpi=300, bbox_inches='tight')
print("Saved: ngrams.png")

pd.DataFrame(top_bigrams_human, columns=['bigram', 'count']).to_csv("../reports/top_bigrams_human.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(top_bigrams_ai, columns=['bigram', 'count']).to_csv("../reports/top_bigrams_ai.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(top_trigrams_human, columns=['trigram', 'count']).to_csv("../reports/top_trigrams_human.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(top_trigrams_ai, columns=['trigram', 'count']).to_csv("../reports/top_trigrams_ai.csv", index=False, encoding="utf-8-sig")

print("\n" + "="*20)
print("LEXICAL ANALYSIS")
print("="*20)

all_tokens_human = [t for tokens in human['tokens'] for t in tokens]
all_tokens_ai = [t for tokens in ai['tokens'] for t in tokens]

function_words_human = [t for t in all_tokens_human if t in stopwords]
function_words_ai = [t for t in all_tokens_ai if t in stopwords]

print(f"\nFunction words:")
print(f"Human: {len(function_words_human)} ({len(function_words_human)/len(all_tokens_human)*100:.2f}%)")
print(f"AI: {len(function_words_ai)} ({len(function_words_ai)/len(all_tokens_ai)*100:.2f}%)")

top_func_human = Counter(function_words_human).most_common(10)
top_func_ai = Counter(function_words_ai).most_common(10)

pd.DataFrame(top_func_human, columns=['word', 'count']).to_csv("../reports/top_function_words_human.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(top_func_ai, columns=['word', 'count']).to_csv("../reports/top_function_words_ai.csv", index=False, encoding="utf-8-sig")

print("\nTop function words (Human):", [x[0] for x in top_func_human[:5]])
print("Top function words (AI):", [x[0] for x in top_func_ai[:5]])

print("\n" + "="*20)
print("EDA COMPLETE!")
print("="*20)
print("\nGenerated files:")
print("- eda_statistics.csv")
print("- distributions.png")
print("- wordclouds.png")
print("- ngrams.png")
print("- top_bigrams_human/ai.csv")
print("- top_trigrams_human/ai.csv")
print("- top_function_words_human/ai.csv")