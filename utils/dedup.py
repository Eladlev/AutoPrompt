import random
from sentence_transformers import SentenceTransformer
import faiss

class Dedup:

    def __init__(self, config):
        self.index = None
        self.xb = None
        self.clusters = None
        self.th = config.get("dedup_threshold", 0.5)
        self.model_name = 'all-MiniLM-L6-v2'

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

        distances, indices = self.index.search(self.xb, len(self.xb))

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

    def dedup(self, records):
        """
        Deduplicate the given dataset.
        input: records - a pandas dataframe with a 'text' column
        output: deduplicated - a pandas dataframe with the deduplicated dataset
        """
        if self.clusters is None:
            self.clusters = self.cluster_data(records)

        ids = [min(cluster) for cluster in self.clusters]
        return records.iloc[sorted(ids)]

    def sample(self, records):
        """
        Sample from a given dataset.
        input: records - a pandas dataframe with a 'text' column
        output: sampled - a pandas dataframe with the sampled dataset
        """
        if self.clusters is None:
            self.clusters = self.cluster_data(records)

        samples = [random.choice(list(cluster)) for cluster in self.clusters]
        return records.iloc[sorted(samples)]
