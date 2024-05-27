from abc import ABC
import numpy as np
import torch
from elsa import ELSA
from plugins.fastcompare.algo.algorithm_base import AlgorithmBase, Parameter, ParameterType


class ELSAWrapper(AlgorithmBase, ABC):
    """
    ELSA algorithm for collaborative filtering.
    Details: https://github.com/recombee/ELSA
    https://dl.acm.org/doi/10.1145/3523227.3551482
    """

    def __init__(self, loader, **kwargs):
        self._ratings_df = loader.ratings_df
        self._loader = loader
        self._all_items = self._ratings_df["item"].unique()
        self._rating_matrix = self._ratings_df.pivot(index="user", columns="item", values="rating").fillna(0).values

        self._items_count = self._rating_matrix.shape[1]
        self._weights = None
        self._factors = kwargs.get("factors", 256)
        self._device = torch.device(kwargs.get("device", "cuda"))
        self._num_epochs = kwargs.get("num_epochs", 5)
        self._batch_size = kwargs.get("batch_size", 128)

        self.model = ELSA(n_items=self._items_count, device=self._device, n_dims=self._factors)
        self.A = None  # for storing the item embeddings

    def fit(self):
        """
        Fit the ELSA model to the rating matrix.
        """
        X_csr = self._get_sparse_interaction_matrix(self._rating_matrix)
        self.model.fit(X_csr, batch_size=self._batch_size, epochs=self._num_epochs)
        self.A = torch.nn.functional.normalize(self.model.get_items_embeddings(), dim=-1).cpu().numpy()

    def predict(self, selected_items, filter_out_items, k):
        """
        Predict top-k items for the user, excluding the filtered-out items.

        Parameters:
        - selected_items: list of items already selected by the user.
        - filter_out_items: list of items to exclude from the prediction.
        - k: number of top items to recommend.

        Returns:
        - top_k_items: list of top-k recommended item indices.
        """
        user_interactions = self._get_user_interactions(selected_items)
        predictions = ((user_interactions @ self.A) @ (self.A.T)) - user_interactions
        predictions[:, filter_out_items] = -np.inf  # Filter out specified items
        top_k_items = np.argsort(predictions, axis=1)[:, -k:][:, ::-1]  # Get top-k items

        return top_k_items.flatten()

    def _get_sparse_interaction_matrix(self, rating_matrix):
        """
        Convert the dense rating matrix to a sparse CSR matrix.
        """
        from scipy.sparse import csr_matrix

        return csr_matrix(rating_matrix)

    def _get_user_interactions(self, selected_items):
        """
        Create a user interaction matrix for the selected items.
        """
        user_interactions = np.zeros((1, self._items_count))
        user_interactions[0, selected_items] = 1
        return user_interactions

    @classmethod
    def name(cls):
        return "ELSA"

    @classmethod
    def parameters(cls):
        return [
            Parameter("factors", ParameterType.INT, 256),
            Parameter(
                "device", ParameterType.STRING, "cpu"
            ),  # "cuda" or "cpu" (for some reason, "cuda" is not working in this case for me)
            Parameter("num_epochs", ParameterType.INT, 5),
            Parameter("batch_size", ParameterType.INT, 128),
        ]
