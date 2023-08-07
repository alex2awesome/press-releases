from retriv import SparseRetriever
from typing import List, Union, Tuple
import numba as nb
import numpy.typing as npt
import numpy as np
from numba.typed import List as TypedList
from retriv.sparse_retriever.sparse_retrieval_models.bm25 import bm25
from retriv.paths import sr_state_path
from numba import njit
from retriv.utils.numba_utils import join_sorted_multi_recursive, unsorted_top_k


@njit(cache=True)
def bm25(
    b: float,
    k1: float,
    term_doc_freqs: nb.typed.List[np.ndarray],
    doc_ids: nb.typed.List[np.ndarray],
    filtered_doc_ids: Union[nb.typed.List[np.ndarray], None],
    relative_doc_lens: nb.typed.List[np.ndarray],
    doc_count: int,
    cutoff: int,
) -> Tuple[np.ndarray]:
    if filtered_doc_ids is None:
        unique_doc_ids = join_sorted_multi_recursive(doc_ids)
    else:
        unique_doc_ids = join_sorted_multi_recursive(filtered_doc_ids)

    scores = np.empty(doc_count, dtype=np.float32)
    scores[unique_doc_ids] = 0.0  # Initialize scores

    for i in range(len(term_doc_freqs)):
        if filtered_doc_ids is None:
            indices = doc_ids[i]
        else:
            indices = filtered_doc_ids[i]
        freqs = term_doc_freqs[i]

        df = np.float32(len(doc_ids[i]))
        idf = np.float32(np.log(1.0 + (((doc_count - df) + 0.5) / (df + 0.5))))

        scores[indices] += idf * (
            (freqs * (k1 + 1.0))
            / (freqs + k1 * (1.0 - b + (b * relative_doc_lens[indices])))
        )

    scores = scores[unique_doc_ids]

    if cutoff < len(scores):
        scores, indices = unsorted_top_k(scores, cutoff)
        unique_doc_ids = unique_doc_ids[indices]

    indices = np.argsort(-scores)

    return unique_doc_ids[indices], scores[indices]



class MySparseRetriever(SparseRetriever):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id_mapping_reverse = {v:k for k, v in self.id_mapping.items()}

    def get_doc_ids_and_term_freqs_filter(self,
                                          query_terms: List[str],
                                          allowed_list: Union[npt.ArrayLike, None]) -> List[nb.types.List]:
        if allowed_list is None:
            return [self.get_doc_ids(query_terms), None, self.get_term_doc_freqs(query_terms)]
        else:
            doc_ids = [self.inverted_index[t]["doc_ids"] for t in query_terms]
            allowed_ids = [np.in1d(doc_ids_i, allowed_list, assume_unique=True) for doc_ids_i in doc_ids]
            filtered_doc_ids = [full_list[b] for full_list, b in zip(doc_ids, allowed_ids)]
            tfs = [self.inverted_index[t]["tfs"] for t in query_terms]
            tfs = [full_list[b] for full_list, b in zip(tfs, allowed_ids)]
            return [TypedList(doc_ids), TypedList(filtered_doc_ids), TypedList(tfs)]

    def search(self, query: str, return_docs: bool = True, cutoff: int = 100,
               include_id_list: Union[List[str], None] = None,
               exclude_id_list: Union[List[str], None] = None,
               ) -> List:
        """Standard search functionality.

        Args:
            query (str): what to search for.
            return_docs (bool, optional): wether to return the texts of the documents. Defaults to True.
            cutoff (int, optional): number of results to return. Defaults to 100.
            include_id_list (list[str], optional): list of doc_ids to include. Defaults to None.
            exclude_id_list (list[str], optional): list of doc_ids to exclude. Defaults to None.

        Returns:
            List: results.
        """

        query_terms = self.query_preprocessing(query)
        if not query_terms:
            return {}
        query_terms = [t for t in query_terms if t in self.vocabulary]
        if not query_terms:
            return {}

        include_list = None
        if include_id_list is not None:
            include_list = [self.id_mapping_reverse[i] for i in include_id_list]
        if exclude_id_list is not None:
            exclude_id_list = set([self.id_mapping_reverse[i] for i in exclude_id_list])
            if include_list is None:
                include_list = self.id_mapping_reverse.values()
            include_list = [i for i in include_list if i not in exclude_id_list]

        doc_ids, filtered_doc_ids, term_doc_freqs = self.get_doc_ids_and_term_freqs_filter(query_terms, include_list)

        if self.model == "bm25":
            unique_doc_ids, scores = bm25(
                term_doc_freqs=term_doc_freqs,
                doc_ids=doc_ids,
                relative_doc_lens=self.relative_doc_lens,
                doc_count=self.doc_count,
                filtered_doc_ids=filtered_doc_ids,
                cutoff=cutoff,
                **self.hyperparams,
            )
        elif self.model == "tf-idf":
            unique_doc_ids, scores = tf_idf(
                term_doc_freqs=term_doc_freqs,
                doc_ids=doc_ids,
                doc_lens=self.doc_lens,
                cutoff=cutoff,
            )
        else:
            raise NotImplementedError()

        unique_doc_ids = self.map_internal_ids_to_original_ids(unique_doc_ids)

        if not return_docs:
            return dict(zip(unique_doc_ids, scores))

        return self.prepare_results(unique_doc_ids, scores)

    @staticmethod
    def load(index_name: str = "new-index"):
        """Load a retriever and its index.

        Args:
            index_name (str, optional): Name of the index. Defaults to "new-index".

        Returns:
            SparseRetriever: Sparse Retriever.
        """

        state = np.load(sr_state_path(index_name), allow_pickle=True)["state"][()]

        se = MySparseRetriever(**state["init_args"])
        se.initialize_doc_index()
        se.id_mapping = state["id_mapping"]
        se.doc_count = state["doc_count"]
        se.inverted_index = state["inverted_index"]
        se.vocabulary = set(se.inverted_index)
        se.doc_lens = state["doc_lens"]
        se.relative_doc_lens = state["relative_doc_lens"]
        se.hyperparams = state["hyperparams"]
        if 'id_mapping_reverse' not in state:
            se.id_mapping_reverse = {v:k for k, v in se.id_mapping.items()}
        else:
            se.id_mapping_reverse = state['id_mapping_reverse']

        state = {
            "init_args": se.init_args,
            "id_mapping": se.id_mapping,
            "doc_count": se.doc_count,
            "inverted_index": se.inverted_index,
            "vocabulary": se.vocabulary,
            "doc_lens": se.doc_lens,
            "relative_doc_lens": se.relative_doc_lens,
            "hyperparams": se.hyperparams,
            "id_mapping_reverse": se.id_mapping_reverse,
        }

        return se


if __name__ == "__main__":

    collection = [
        {"id": "doc_1", "text": "Generals gathered in their masses"},
        {"id": "doc_2", "text": "Just like witches at black masses"},
        {"id": "doc_3", "text": "Evil minds that plot destruction"},
        {"id": "doc_4", "text": "Sorcerer of death's construction"},
    ]

    # se = MySparseRetriever.load("new-index")
    se = MySparseRetriever('new-index')
    se.index(collection)
    print(se.search("witches masses"))
    print(se.search("witches masses", include_id_list=["doc_2", "doc_3", "doc_4"]))