import pandas as pd

gaia_benchmark_path = ''
df = pd.read_json(gaia_benchmark_path, lines=True)
num_samples = 10
initial_data = df.sample(num_samples)
q = initial_data['Question'].tolist()
f = initial_data['Final answer'].tolist()
fn = initial_data['file_name'].tolist()
for i,(ff,qq) in enumerate(zip(fn,q)):
    if not ff == '':
        q[i] = qq+ '\nAttached file:\n' + './dump/files/' + ff

data = {
    'id': list(range(num_samples)),
    'text': q,
    'prediction': [None]*num_samples,
    'annotation': f,
    'metadata': [None]*num_samples,
    'score': [None]*num_samples,
    'batch_id': [0]*num_samples,
    'Level': initial_data['Level'].tolist(),
}


df = pd.DataFrame(data)
df.to_csv('dump/root/dataset.csv', index=False)