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
        self._items_df = loader.items_df
        self._loader = loader
        self._all_items = self._ratings_df["item"].unique()
        self._rating_matrix = self._ratings_df.pivot(index="user", columns="item", values="rating").fillna(0).values
        self._item_years = self._items_df.set_index("item_id")["year"].to_dict()
        self._items_count = self._rating_matrix.shape[1]
        self._weights = None

        self._factors = kwargs.get("factors", 256)
        self._device = torch.device(kwargs.get("device", "cpu"))
        if kwargs.get("device") == "cuda" and not torch.cuda.is_available():
            print("CUDA is not available, using CPU instead.")
            self._device = torch.device("cpu")

        self._num_epochs = kwargs.get("num_epochs", 5)
        self._batch_size = kwargs.get("batch_size", 128)
        self._postprocess_diversity = kwargs.get("postprocess-diversity", True)
        self._postprocess_novelty = kwargs.get("postprocess-novelty", True)
        self._postprocess_serendipity = kwargs.get("postprocess-serendipity", True)
        self._diversity_threshold = kwargs.get("diversity_threshold", 0.5)
        self._novelty_weight = kwargs.get("novelty_weight", 0.9)
        self._serendipity_weight = kwargs.get("serendipity_weight", 0.5)

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

        x = 5
        top_xk_items = np.argsort(predictions, axis=1)[:, -x * k :][:, ::-1]  # Get top-xk items

        if self._postprocess_diversity:
            top_xk_items = self._apply_diversity(top_xk_items)
        if self._postprocess_novelty:
            top_xk_items = self._apply_novelty(top_xk_items)
        if self._postprocess_serendipity:
            top_xk_items = self._apply_serendipity(top_xk_items)

        if top_xk_items.size < k:
            top_k_items = np.argsort(predictions, axis=1)[:, -k:][:, ::-1]
            top_xk_items = self._add_items_if_needed(top_xk_items, top_k_items, k)

        return top_xk_items.flatten()[:k]

    def _add_items_if_needed(self, top_xk_items, top_k_items, k):
        """
        Add items from top-k that are not in top-xk to ensure at least k items.

        Parameters:
        - top_xk_items: list of top-x*k recommended item indices.
        - top_k_items: list of top-k recommended item indices.
        - k: number of top items to recommend.

        Returns:
        - final_items: list of top-k recommended item indices.
        """
        top_xk_set = set(top_xk_items.flatten())
        top_k_list = top_k_items.flatten()
        additional_items = [item for item in top_k_list if item not in top_xk_set]

        final_items = top_xk_items.flatten().tolist()
        while len(final_items) < k:
            final_items.append(additional_items.pop(0))

        return np.array(final_items).reshape(1, -1)

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
        The check works by calculating the dot product of the item embeddings.

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

    def _apply_novelty(self, top_k_items):
        """
        Apply novelty to the recommended items based on the year.
        Novelty is calculated as the difference between the latest year found in the dataset and the item year.

        Parameters:
        - top_k_items: list of top-k recommended item indices.

        Returns:
        - novel_items: list of top-k recommended item indices with novelty applied.
        """
        latest_year = max(self._item_years.values())
        novel_scores = np.array([latest_year - self._item_years.get(item, latest_year) for item in top_k_items[0]])
        novelty_weighted_scores = novel_scores * self._novelty_weight
        novel_items = top_k_items[0][np.argsort(-novelty_weighted_scores)]
        return novel_items.reshape(1, -1)

    def _apply_serendipity(self, top_k_items):
        """
        Apply serendipity to the recommended items.

        Parameters:
        - top_k_items: list of top-k recommended item indices.

        Returns:
        - serendipitous_items: list of top-k recommended item indices with serendipity applied.
        """
        serendipity_scores = np.zeros(len(top_k_items[0]))
        for idx, item in enumerate(top_k_items[0]):
            serendipity_scores[idx] = self._calculate_serendipity(item)
        serendipity_weighted_scores = serendipity_scores * self._serendipity_weight
        serendipitous_items = top_k_items[0][np.argsort(-serendipity_weighted_scores)]
        return serendipitous_items.reshape(1, -1)

    def _calculate_serendipity(self, item):
        """
        Calculate the serendipity score for an item.
        Serendipity is calculated as the dissimilarity to the user's selected items.

        Parameters:
        - item: item index to calculate serendipity score.

        Returns:
        - serendipity_score: calculated serendipity score.
        """
        selected_items_embedding = np.mean(self.A, axis=0)
        item_embedding = self.A[item]
        serendipity_score = 1 - np.dot(selected_items_embedding, item_embedding) / (
            np.linalg.norm(selected_items_embedding) * np.linalg.norm(item_embedding)
        )
        return serendipity_score

    @classmethod
    def name(cls):
        return "ELSA"

    @classmethod
    def parameters(cls):
        return [
            Parameter("factors", ParameterType.INT, 256, help="Number of latent factors for the model."),
            Parameter(
                "device", ParameterType.STRING, "cpu", help="Device to run the model on ('cuda' or 'cpu')."
            ),  # this could be improved with ParemeterType.OPTIONS, but there is no documentation
            Parameter("num_epochs", ParameterType.INT, 5, help="Number of epochs to train the model."),
            Parameter("batch_size", ParameterType.INT, 128, help="Batch size for training."),
            Parameter(
                "postprocess-diversity",
                ParameterType.BOOL,
                True,
                help="Enable or disable diversity in recommendations.",
            ),
            Parameter(
                "postprocess-novelty",
                ParameterType.BOOL,
                True,
                help="Enable or disable novelty in recommendations.",
            ),
            Parameter(
                "postprocess-serendipity",
                ParameterType.BOOL,
                True,
                help="Enable or disable serendipity in recommendations.",
            ),
            Parameter(
                "diversity_threshold",
                ParameterType.FLOAT,
                0.5,
                help="Threshold for diversity in recommendations. Higher values mean less diversity.",
            ),
            Parameter(
                "novelty_weight",
                ParameterType.FLOAT,
                0.9,
                help="Weight for novelty in recommendations. Higher values give more importance to newer items.",
            ),
            Parameter(
                "serendipity_weight",
                ParameterType.FLOAT,
                0.5,
                help="Weight for serendipity in recommendations. Higher values give more importance to unexpected items.",
            ),
        ]
