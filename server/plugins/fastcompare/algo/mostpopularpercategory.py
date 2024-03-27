from abc import ABC

import sys
import numpy as np
import pandas as pd
from plugins.fastcompare.algo.algorithm_base import (
    AlgorithmBase,
    Parameter,
    ParameterType,
)


class MostPopularPerCategory(AlgorithmBase, ABC):
    """Implementation of MostPopularPerCategory algorithm.

    Generating recommendation of length k:
    If k <= |categories| then sample k unique categories at random and take most popular item for each of them
    Otherwise sample with replacement and do the same
    """

    def __init__(self, loader, **kwargs):
        self._ratings_df = loader.ratings_df
        self._loader = loader
        self._all_items = self._ratings_df.item.unique()
        self._categories = list(self._loader.get_all_categories())

        self._rating_matrix = (
            self._loader.ratings_df.pivot(index="user", columns="item", values="rating").fillna(0).values
        )

        self._items_count = np.shape(self._rating_matrix)[1]
        self._weights = None

    # One-time fitting of the algorithm for a predefined number of iterations
    def fit(self):
        # fit is not needed for this algorithm
        pass

    # Predict for the user
    def predict(self, selected_items, filter_out_items, k):
        categories = []

        for item in selected_items:
            item_categories = self._loader.get_item_index_categories(item)
            categories.extend(item_categories)

        selected_categories = list(set(categories))

        # If k <= |categories| then sample k unique categories at random and take most popular item for each of them
        # Otherwise sample with replacement and do the same
        """
        if k <= len(categories):
            selected_categories = np.random.choice(self._categories, k, replace=False)
        else:
            selected_categories = np.random.choice(selected_categories, k, replace=True)
        """

        result = []

        categories_items_dict = {category: [] for category in selected_categories}

        for item in self._all_items:
            item_categories = self._loader.get_item_index_categories(item)
            for category in item_categories:
                if category in selected_categories:
                    categories_items_dict[category].append(item)

        # now rate the items in each category
        for category, items in categories_items_dict.items():
            # the ratings are in format: userId, movieId, rating, timestamp
            # we need to find the most popular item in the category
            # we can do this by calculating the average rating for each item in the category
            # but make sure every item has at least 5 ratings

            print(self._ratings_df, file=sys.stderr)

            items_ratings = self._ratings_df[self._ratings_df.item_id.isin(items)].groupby("item_id").rating.mean()
            if len(items_ratings) == 0:
                continue

            most_popular_item = items_ratings.idxmax()
            result.append(most_popular_item)

            if len(items_ratings) < 5:
                print(f"Category {category} has less than 5 items", file=sys.stderr)
                print(items_ratings, file=sys.stderr)
            else:
                print(f"Category {category}", file=sys.stderr)
                print(items_ratings[:5], file=sys.stderr)
                print(most_popular_item, file=sys.stderr)
                print(items_ratings.max(), file=sys.stderr)

        return result[:k]

    @classmethod
    def name(cls):
        return "MostPopularPerCategory"

    @classmethod
    def parameters(cls):
        return []
