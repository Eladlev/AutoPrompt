import pandas as pd
def format_text(row):
   return f"{{{row['Utterance']}}}#{{{row['Article']}}}"


df = pd.read_csv('./classifier_dataset_binary.csv')
# Use forward fill to fill missing values in the 'Utterance' column
df['text'] = df.apply(format_text, axis=1)
new_df = pd.DataFrame({
   'id': df.index,
   'text': df['text'],
   'prediction': 'Irrelevant',  # Setting 'prediction' to 'No' for all rows
   'annotation': df['Label'],  # Copying 'Label' to 'annotation'
   'metadata': '',  # Setting 'metadata' to empty
   'score': '',  # Setting 'score' to empty
   'batch_id': 0  # Setting 'batch_id' to 0 for all rows
})
new_df.to_csv('./formatted_classifier_dataset_binary.csv', index=False)
