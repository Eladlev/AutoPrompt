import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support

# Load your dataset
df = pd.read_csv('dump/dataset.csv')

# Ensure the annotations and predictions are in categorical form
categories = ['Relevant', 'Irrelevant', 'Partially']
df['annotation'] = pd.Categorical(df['annotation'], categories=categories, ordered=False).astype(str)
df['prediction'] = pd.Categorical(df['prediction'], categories=categories, ordered=False).astype(str)

# Compute the confusion matrix
conf_matrix = confusion_matrix(df['annotation'], df['prediction'], labels=categories)
print("Confusion Matrix:")
print(conf_matrix)

# Compute Accuracy
accuracy = accuracy_score(df['annotation'], df['prediction'])
print("\nAccuracy: {:.2f}%".format(accuracy * 100))

# Compute Precision, Recall, and F1-score
precision, recall, f1_score, _ = precision_recall_fscore_support(df['annotation'], df['prediction'], labels=categories)

# Display the results
for i, category in enumerate(categories):
    print(f"\nCategory: {category}")
    print(f"Precision: {precision[i]:.2f}")
    print(f"Recall: {recall[i]:.2f}")
    print(f"F1-Score: {f1_score[i]:.2f}")