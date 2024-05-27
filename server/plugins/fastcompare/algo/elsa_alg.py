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
        self._device = torch.device(kwargs.get("device", "cpu"))
        if kwargs.get("device") == "cuda" and not torch.cuda.is_available():
            print("CUDA is not available, using CPU instead.")
            self._device = torch.device("cpu")
        self._num_epochs = kwargs.get("num_epochs", 5)
        self._batch_size = kwargs.get("batch_size", 128)
        self._postprocess = kwargs.get("postprocess", True)
        self._diversity_threshold = kwargs.get("diversity_threshold", 0.9)

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

        if self._postprocess:
            top_k_items = self._apply_diversity(top_k_items)

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

    def _apply_diversity(self, top_k_items):
        """
        Apply diversity to the recommended items.

        Parameters:
        - top_k_items: list of top-k recommended item indices.

        Returns:
        - diversified_items: list of diversified top-k recommended item indices.
        """
        diversified_items = []
        for item in top_k_items[0]:
            if not self._is_too_similar(item, diversified_items):
                diversified_items.append(item)
                if len(diversified_items) >= len(top_k_items[0]):
                    break
        return np.array(diversified_items).reshape(1, -1)

    def _is_too_similar(self, item, diversified_items):
        """
        Check if the item is too similar to any of the already selected items.

        Parameters:
        - item: item index to check.
        - diversified_items: list of already selected diversified items.

        Returns:
        - is_similar: boolean indicating if the item is too similar.
        """
        for div_item in diversified_items:
            similarity = np.dot(self.A[item], self.A[div_item])
            if similarity > self._diversity_threshold:
                return True
        return False

    @classmethod
    def name(cls):
        return "ELSA"

    @classmethod
    def parameters(cls):
        return [
            Parameter("factors", ParameterType.INT, 256, help="Number of latent factors for the model."),
            Parameter("device", ParameterType.STRING, "cpu", help="Device to run the model on ('cuda' or 'cpu')."),
            # Parameter("device", ParameterType.OPTIONS, "cpu", "cuda"), # TODO: try this, but I don't know the options format
            Parameter("num_epochs", ParameterType.INT, 5, help="Number of epochs to train the model."),
            Parameter("batch_size", ParameterType.INT, 128, help="Batch size for training."),
            Parameter("postprocess", ParameterType.BOOL, True, help="Enable or disable postprocessing for diversity."),
            Parameter(
                "diversity_threshold",
                ParameterType.FLOAT,
                0.9,
                help="Threshold for diversity in recommendations. Higher values mean less diversity.",
            ),
        ]
