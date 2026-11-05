import os

import faiss
import numpy as np

INDEX_FILENAME = "kNN.index"
LABELS_FILENAME = "idx2class.npz"
EPSILON = 1e-4


class FaissClassifier:
    """Retrieval database: L2-normalised embeddings with one class label each."""

    def __init__(self, database_dir: str, dim: int):
        index_path = os.path.join(database_dir, INDEX_FILENAME)
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"No retrieval database at {index_path}. Build one with "
                "build_database.py, see GETTING_STARTED.md."
            )
        self.index = faiss.read_index(index_path)
        self.index_to_class = np.load(
            os.path.join(database_dir, LABELS_FILENAME)
        )["idx_to_class"]
        if self.index.d != dim:
            raise ValueError(
                f"{database_dir} holds {self.index.d}-d embeddings but the "
                f"feature extractor produces {dim}-d ones."
            )
        # Search scores are summed as similarities in compute_class_scores,
        # which would silently invert the weighting for a distance metric.
        if self.index.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise ValueError(
                f"The index in {database_dir} must use the inner product "
                "metric: rebuild it with build_database.py."
            )
        if self.index.ntotal != len(self.index_to_class):
            raise ValueError(
                f"{database_dir} has {self.index.ntotal} embeddings but "
                f"{len(self.index_to_class)} labels."
            )
        self.max_label = int(self.index_to_class.max())

    def compute_class_scores(
        self, queries: np.ndarray, k: int, num_classes: int
    ) -> np.ndarray:
        """
        Sum the cosine similarities of the k nearest neighbours per class, the
        exponent of Eq. (3).

        Args:
            queries: embeddings of shape [N, dim]
            k: number of neighbours to retrieve
            num_classes: size of the vocabulary of the model
        Returns:
            scores of shape [N, num_classes]
        """
        if self.max_label >= num_classes:
            raise ValueError(
                f"The database has labels up to {self.max_label} but the "
                f"vocabulary has {num_classes} classes: build the database "
                "from the dataset being evaluated."
            )
        queries = queries / (
            np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8
        )
        similarities, neighbours = self.index.search(queries, k)
        scores = np.zeros((len(queries), num_classes))
        # HNSW pads with -1 when it finds fewer than k neighbours.
        rows, cols = np.nonzero(neighbours >= 0)
        np.add.at(
            scores,
            (rows, self.index_to_class[neighbours[rows, cols]]),
            similarities[rows, cols],
        )
        return scores + EPSILON
