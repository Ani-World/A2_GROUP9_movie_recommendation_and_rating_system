import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import json
import os # <-- IMPORT OS

# --- Define Correct Paths ---
# Assumes this script is in the 'Backend' folder, like app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'datasets', 'movies_data_processed.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True) # Ensure models directory exists

# Load CSV
try:
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Loaded {DATA_PATH}")
except FileNotFoundError:
    print(f"❌ ERROR: Data file not found at {DATA_PATH}")
    print("Please make sure 'movies_data_processed.csv' is in the 'datasets' folder.")
    exit()


# Expect 'apriori_items' as comma-separated strings of genres (or items)
if 'apriori_items' not in df.columns:
    print("❌ ERROR: 'apriori_items' column not in CSV. Cannot generate rules.")
    exit()

transactions = df['apriori_items'].dropna().apply(lambda x: [item.strip().strip("'\"") for item in x.split(',')])

# Convert to one-hot encoded DataFrame
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_ohe = pd.DataFrame(te_ary, columns=te.columns_)

# Run Apriori
frequent_itemsets = apriori(df_ohe, min_support=0.0005, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)

# Keep rules with a single consequent (simpler for recommendations)
rules = rules[rules['consequents'].apply(lambda x: len(x) == 1)]

# Map rules to dict: consequent -> list of textual rules
apriori_rules = {}
for _, row in rules.iterrows():
    consequent = list(row['consequents'])[0]  # this is a string now
    antecedent = ', '.join(list(row['antecedents']))
    text = f"Because users who liked {antecedent} also liked {consequent}."
    if consequent not in apriori_rules:
        apriori_rules[consequent] = []
    apriori_rules[consequent].append(text)

# --- Save rules to JSON in the correct models folder ---
output_path = os.path.join(MODEL_DIR, 'apriori_rules.json') # <-- Use correct path
with open(output_path, 'w') as f:
    json.dump(apriori_rules, f, indent=2)

print(f"✅ Generated {len(apriori_rules)} apriori rules and saved to {output_path}")