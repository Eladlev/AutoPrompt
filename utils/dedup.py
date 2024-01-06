import random
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd


class Dedup:

    def __init__(self, config=None):
        self.index = None
        self.xb = None
        self.clusters = None
        self.th = (config or {}).get("dedup_threshold", 0.5)
        self.model_name = (config or {}).get("embeddings_model", 'all-MiniLM-L6-v2')

    def copy(self):
        return Dedup(
            {"dedup_threshold": self.th,
             "embeddings_model": self.model_name}
        )

    def generate_embeddings(self, texts):
        """
        Generate embeddings for the given texts using the SentenceTransformer model.
        """
        model = SentenceTransformer(self.model_name)
        embeddings = model.encode(texts, show_progress_bar=True)
        return embeddings

    def build_index(self, records):
        """
        Build the FAISS index for the given dataset.
        input: records - a pandas dataframe with a 'text' column
        output: index - the FAISS index
                embeddings - the embeddings of the dataset
        """
        # Generate embeddings for the dataset
        embeddings = self.generate_embeddings(records['text'].tolist())

        # Build the FAISS index
        embeddings_dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(embeddings_dim)
        index.add(embeddings)
        return index, embeddings

    def cluster_data(self, records):
        """
        Cluster the given dataset.
        input: records - a pandas dataframe with a 'text' column
        output: clusters - a list of clusters, where each cluster is a set of indices
        """

        if self.index is None:
            self.index, self.xb = self.build_index(records)

        distances, indices = self.index.search(self.xb, 30) #TODO: dereive it from the batch size

        clusters = []
        visited = set()

        for i in range(len(self.xb)):
            if i in visited:
                continue

            # Find neighbors and create a new cluster
            neighbors = [idx for idx, distance in zip(indices[i], distances[i]) if distance <= self.th]
            new_cluster = {i}

            # Add all neighbors to the new cluster
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_cluster.add(neighbor)

            clusters.append(new_cluster)
        return clusters

    def sample(self, records: pd.DataFrame, operation_function=random.choice):
        """
        Sample the given dataset.
        input: records - a pandas dataframe with a 'text' column
               operation_function - a function that receives a cluster and returns an index
        output: a pandas dataframe with the sampled records
        """

        if not callable(operation_function):
            raise ValueError("The 'operation_function' must be a callable function.")

        if self.clusters is None:
            self.clusters = self.cluster_data(records)

        samples = [operation_function(list(cluster)) for cluster in self.clusters]
        return records.iloc[sorted(samples)]
