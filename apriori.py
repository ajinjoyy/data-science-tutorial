import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# ✅ Step 1: Create a Dataset
# Simulating transactions with one-hot encoding
data = {
    'Milk': [1, 0, 1, 1, 0],
    'Bread': [1, 1, 1, 0, 1],
    'Butter': [1, 1, 0, 1, 0],
}
df = pd.DataFrame(data)

print("Transaction Dataset:")
print(df)

# ✅ Step 2: Apply Apriori Algorithm to Find Frequent Itemsets
# - min_support: Minimum support threshold (e.g., 0.4 means the itemset should appear in 40% of transactions)
# - use_colnames: Ensures item names are used instead of column indexes
frequent_itemsets = apriori(df, min_support=0.4, use_colnames=True)

print("\nFrequent Itemsets:")
print(frequent_itemsets)

# ✅ Step 3: Generate Association Rules
# - metric='confidence': Generates rules based on confidence
# - min_threshold=0.5: Minimum confidence level for rules
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.5)

print("\nGenerated Association Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# ✅ Step 4: Interpretation
# The output shows which items are frequently bought together and the strength of their relationship
