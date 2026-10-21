"""Python module for tagging documents with taxonomy terms
Author: Andreas Hamm (andreas.hamm@dlr.de)
Version: 1.0.0
Copyright: DLR
License: MIT
"""


from collections import defaultdict
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union
import nltk
from nltk.tokenize import sent_tokenize
import pandas as pd
import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import ast

# Maps two-letter ISO codes to the punkt_tab language directory names NLTK's
# Punkt tokenizer loads. These must match the directory names on disk
# (e.g. "german", "english"): punkt.py does a plain ``find("tokenizers/punkt_tab/{lang}/")``
# with no case normalization, so capitalization must match the filesystem
# (which is fine on Windows but would fail on case-sensitive Linux/macOS).
# Only models actually installed as ``punkt_tab`` will work.
_ISO_TO_NLTK_LANGUAGE = {
    "en": "english",
    "de": "german",
}


def _ensure_punkt(language: str = "english") -> bool:
    """Ensures the punkt_tab tokenizer for ``language`` is available.

    NLTK only downloads punkt_tab lazily and raises ``LookupError`` otherwise.
    On first use it attempts a silent download of ``punkt_tab`` and retries.

    Args:
        language (str): The punkt_tab language directory name, e.g. ``"english"`` or ``"german"``.

    Returns:
        bool: ``True`` if the tokenizer is available afterwards, ``False`` otherwise.
    """
    try:
        nltk.data.find(f"tokenizers/punkt_tab/{language}")
        return True
    except LookupError:
        try:
            return bool(nltk.download("punkt_tab", quiet=True))
        except Exception:
            return False

class ConceptTaxonomy:
    """Class for working with a concept hierarchy including concept embeddings

    Attributes:
        default_lang (str): Default language (two-letter ISO code of a supported language)
        tg (networkx.DiGraph): Graph defining the concept hierarchy. The concepts are nodes with attributes
                               - ``key``
                               - ``label_{isolang}``
                               - ``description_{isolang}``
                               where ``isolang`` is the two-letter ISO code of each supported language (e.g., ``"en"``, ``"de"``).
                               The edges point from concept nodes to their parents.
        root (str): Key of the formal root node.
        transformer (sentence_transformers.SentenceTransformer): SentenceTransformer model used for initial embeddings (only set if ``transformer`` was passed).
        classifier (None or transformers.pipeline): Optionally a Huggingface zero-shot-classification pipeline.
        desc_lang (str): Current language of concept descriptions (two-letter ISO code of a supported language).
        desc_concepts (List[Tuple[str, str, str]]): List of ``(key, label, sentence)`` triples for all concepts, with labels and
                                          descriptive sentences in the current language.
        model (sentence_transformers.SentenceTransformer): SentenceTransformer model used for current embeddings (only set if ``transformer`` was passed).
        emb_concepts (Optional[numpy.ndarray]): Row-normalized embedding vectors for all concepts in ``desc_concepts``.
        token (None or str): Optionally Huggingface Access token.
    """

    def __init__(
        self,
        taxgraph: nx.DiGraph,
        root: str,
        long_descr: bool = True,
        transformer: Optional[Union[str, SentenceTransformer]] = None,
        default_lang: str = "en",
        classifier: Optional[str] = None,
        token: Optional[str] = None,
    ):
        """Constructs a ``ConceptTaxonomy``, optionally computing an initial embedding.

        Args:
            taxgraph (networkx.DiGraph): Graph defining the concept hierarchy. The concepts are nodes with attributes
                                         - ``key``
                                         - ``label_{isolang}``
                                         - ``description_{isolang}``
                                         where ``isolang`` is the two-letter ISO code of each supported language (e.g., ``"en"``, ``"de"``)
            root (str): Key of the formal root node. Must be a node in ``taxgraph``.
            long_descr (bool, optional): Whether to include the concept descriptions in the concept comparison sentences. Defaults to ``True``.
            transformer (Union[str, SentenceTransformer], optional): SentenceTransformer model (or its name) used for initial embeddings. Defaults to ``None``.
            default_lang (str, optional): Default language (two-letter ISO code of a supported language). Defaults to ``"en"``.
            classifier (str, optional): Optionally a Huggingface zero-shot-classification model name. Defaults to ``None``.
            token (str, optional): Huggingface access token. Defaults to ``None``.

        Raises:
            ValueError: If ``root`` is not a node in ``taxgraph``.
        """
        self.default_lang = default_lang
        self.tg = taxgraph
        self.root = root
        self.token = token
        # Ensure the punkt_tab sentence tokenizer for the default language is
        # available, since score_concepts_sentencewise depends on it.
        _ensure_punkt(_ISO_TO_NLTK_LANGUAGE.get(self.default_lang, "english"))
        if long_descr:
            self.describe_concepts()
        else:
            self.describe_concepts(long=False)
        if transformer:
            self.transformer = transformer
            self.embed_concepts(transformer)
        if classifier:
            self.classifier = pipeline("zero-shot-classification", model=classifier)
        else:
            self.classifier = None

    def from_df(
        self,
        df_concepts: pd.DataFrame,
        rename_columns: Optional[Dict[str, str]] = None,
    ) -> "ConceptTaxonomy":
        """Reads in new concepts from a dataframe.

        Args:
            df_concepts (pandas.DataFrame): Dataframe. Each row contains a node to be added to the graph defining the concept hierarchy.
                                            The dataframe has to have the columns
                                            - ``key`` for the concept key
                                            - ``parents``, a list with the parent keys
                                            - ``label_xx`` with ``xx`` the default language for the concept label
                                            - ``description_xx`` with ``xx`` the default language for the concept description (might be empty)
                                            and optionally further ``label_yz``, ``description_yz`` for other supported languages, and also other attributes
            rename_columns (dict, optional): Dictionary for optionally renaming column headings. Defaults to ``{}``.

        Returns:
            ConceptTaxonomy: This instance, to allow method chaining.
        """
        if rename_columns is None:
            rename_columns = {}
        if rename_columns:
            df_concepts = df_concepts.rename(columns=rename_columns)
        if len(set(df_concepts.keys()) & set(["key", f"label_{self.default_lang}", f"description_{self.default_lang}", "parents"])) < 4:
            self.describe_concepts()
            if self.transformer:
                self.embed_concepts()
            return self
        for index, row in df_concepts.iterrows():
            self.tg.add_node(row["key"], **{k: v for k, v in row.items() if k not in ["key", "parents"]})
        for index, row in df_concepts.iterrows():
            for p in ast.literal_eval(row["parents"]):
                self.tg.add_edge(row["key"], p)
        # Refresh the derived concept/embedding caches, since the nodes above
        # changed but the caches from __init__ did not.
        self.describe_concepts()
        if self.transformer:
            self.embed_concepts()
        return self

    def to_df(self) -> pd.DataFrame:
        """Writes the concept hierarchy into a dataframe.

        Returns:
            pandas.DataFrame: Dataframe. Each row contains one concept. The columns are ``key``, all attributes of the concept nodes, and ``parents``.
        """
        records = []
        for conc, attr in self.tg.nodes.items():
            records.append({
                "key": conc,
                **attr,
                "parents": [p for p in self.tg.predecessors(conc)]
            })
        return pd.DataFrame(records)


    def describe_concepts(
        self,
        lang: Optional[str] = None,
        intro: Optional[str] = None,
        long: bool = True,
    ) -> None:
        """Fills the attributes :pyattr:`desc_concepts` (a list of ``(key, label, sentence)`` triples) and :pyattr:`desc_lang`.

        Args:
            lang (str, optional): Two-letter ISO code for language to use for concept names and descriptions. Defaults to ``None``, which means ``self.default_lang``.
            intro (str, optional): Phrase for forming concept comparison sentences. Defaults to ``'This text is about '`` (which gets translated into ``'Dieser Text handelt von '`` in German).
            long (bool, optional): Whether to include the concept descriptions in the concept comparison sentences. Defaults to ``True``.
        """
        if lang is None:
            lang = self.default_lang
        if intro is None:
            if lang == "de":
                intro = "Dieser Text handelt von "
            elif lang == "en":
                intro = "This text is about "
            else:
                intro = ""
        self.desc_lang = lang
        self.desc_concepts = []
        for ckey, cattrib in self.tg.nodes.items():
            label = cattrib.get(f"label_{lang}", cattrib.get("label_en"))
            if label is None:
                continue
            concept_sent = intro + label
            if long:
                description = cattrib.get(f"description_{lang}", cattrib.get("description_en"))
                if description:
                    concept_sent += " (" + description + ")"
            self.desc_concepts.append((ckey, label, concept_sent))

    def embed_concepts(self, model: Optional[SentenceTransformer] = None) -> None:
        """Fills the attribute :pyattr:`emb_concepts` (the embedding of all concept comparison sentences) and sets :pyattr:`model`.
        On large taxonomies, this method will have a considerable execution time.

        Args:
            model (sentence_transformers.SentenceTransformer, optional): SentenceTransformer model to be used for embeddings. Defaults to ``None``, which reuses ``self.transformer``.
        """
        if model is None:
            model = self.transformer
        self.model = model
        if len(self.desc_concepts) == 0:
            self.emb_concepts = []
            return
        emb_sents = model.encode([x[2] for x in self.desc_concepts], convert_to_numpy=True)
        self.emb_concepts = emb_sents / np.linalg.norm(emb_sents, axis=1)[:, None]

    def get_concepts_in_text(
        self,
        text: str,
        top_k: int = 10,
        threshold: float = 0.3,
        sim_func: Optional[Callable[[Any, Any], Any]] = None,
    ) -> List[Tuple[str, str, float]]:
        """Similarity search for taxonomy concepts in a given text.
        Uses either a zero-shot classification transformers pipeline (given as optional class attribute ``classifier``) or, by default, cosine similarity in the SentenceTransformer model's embedding space.

        Args:
            text (str): Text.
            top_k (int, optional): Maximum number of concepts returned. Defaults to ``10``.
            threshold (float, optional): Minimum similarity threshold. Defaults to ``0.3``.
            sim_func (Callable[[Tensor, Tensor], Tensor], optional): Similarity function passed to ``util.semantic_search``. Defaults to ``None``, which means cosine similarity.

        Returns:
            list: List of found concepts as ``(concept key, concept name, similarity score)`` triples, sorted by descending score.
        """
        if len(self.desc_concepts) == 0:
            return []
        if self.classifier:
            premises = [cd[2] for cd in self.desc_concepts]
            clion = self.classifier(text, premises, hypothesis_template="{}", multi_class=True)
            scored = [(cd[0], cd[1], score)
                      for cd, score in zip(self.desc_concepts, clion["scores"])
                      if score > threshold]
            return sorted(scored, key=lambda z: -z[2])[:top_k]
        emb_sent = self.model.encode(text, convert_to_numpy=True)
        if sim_func is not None:
            return [(self.desc_concepts[x["corpus_id"]][0], self.desc_concepts[x["corpus_id"]][1], x["score"])
                    for x in util.semantic_search(emb_sent, self.emb_concepts, top_k=top_k, score_function=sim_func)[0] if x["score"] > threshold]
        return [(self.desc_concepts[x["corpus_id"]][0], self.desc_concepts[x["corpus_id"]][1], x["score"])
                for x in util.semantic_search(emb_sent, self.emb_concepts, top_k=top_k)[0] if x["score"] > threshold]

    def get_parent_concepts(
        self,
        child: str,
        top_parent: Optional[str] = None,
        lang: Optional[str] = None,
    ) -> Iterator[List[Union[str, Tuple[str, str, str]]]]:
        """Returns paths of predecessors of a given child concept, up back to a ``top_parent``.

        Args:
            child (str): Key of child concept.
            top_parent (str, optional): Key of earliest ancestor. Defaults to ``None``, in which case it is the root node of the hierarchy.
            lang (str, optional): Two-letter ISO code for language to use for concept names and descriptions. Defaults to ``None``, in which case just the concept keys are returned.

        Returns:
            list: An iterator of paths to ``top_parent``, each path being a list of concepts. If ``lang`` is ``None``, each path is a list of concept keys; otherwise each concept is a ``(key, name, description)`` triple.
        """
        if top_parent is None:
            top_parent = self.root
        paths = nx.all_simple_paths(self.tg, target=top_parent, source=child)
        if lang is None:
            return paths
        return [[(k, self.tg.nodes[k][f"label_{lang}"], self.tg.nodes[k][f"description_{lang}"]) for k in p] for p in paths]

    def show_concepts(
        self,
        concepts: List[List[str]],
        lang: Optional[str] = None,
        scores: Optional[List[float]] = None,
    ) -> List[List[Union[Tuple[str, str], Tuple[str, str, float]]]]:
        """Convenience function for producing output of concept results.

        Args:
            concepts (list): List of hierarchy paths leading to concepts, each path represented as a list of concept keys.
            lang (str, optional): Two-letter ISO code for language to use for concept names and descriptions. Defaults to ``None``, which means ``self.default_lang``.
            scores (list, optional): List of scores along a hierarchical concept path. Defaults to ``None``.

        Returns:
            list: List of lists of ``(concept name, concept key)`` pairs, or ``(concept name, concept key, aggregated score)`` triples when ``scores`` is given.
        """
        if lang is None:
            lang = self.default_lang
        if scores is None:
            return [
                [(self.tg.nodes[ck].get(f"label_{lang}", ck), ck) for ck in path]
                for path in concepts
            ]
        return [
            [(self.tg.nodes[ck].get(f"label_{lang}", ck), ck, sc) for ck in path]
            for path, sc in zip(concepts, scores)
        ]


class TaxedDocument:
    """Class for working with a document to detect related concepts of a taxonomy in it

    Attributes:
        taxonomy (ConceptTaxonomy): The taxonomy from which the concept tags to the document are taken
        doc_text (str): Text of the document
        overall_concepts (Dict[str, float]): Concept scores for whole text (Key: Concept key, Value: Score)
        sentences_concepts (Dict[str, List[float]]): Sentencewise concept scores (Key: Concept key, Value: List of sentence scores)
        aggregated_concepts (Dict[str, float]): Aggregated concept scores (Key: Concept key, Value: Score)
        hierarchy_distribution (Dict[int, Dict[str, float]]): Aggregated concept scores summed up along hierarchical upward paths.
                                                               Key: Hierarchy level, Value: Dict with Key: Concept key, Value: Summed-up score divided by path length
        hierarchy_restriction (Dict[int, List[str]]): Consistency-restricted concepts on hierarchical levels.
                                                       Key: Hierarchical level, Value: Restricted list of concepts that make up the required percentage of scores
        top_k (int): Current number of concepts retained when checking similarity to text
        threshold (float): Current minimum similarity for considering concepts
        weight_dev (Dict[int, float]): Dictionary for setting sentence weights different from 1 (Key: sentence number, Value: Sentence weight)
        kparm (float): Current saturation parameter for aggregating similarities
    """

    def __init__(self, taxonomy: ConceptTaxonomy, doc_text: str):
        """Constructor for TaxedDocument.

        Args:
            taxonomy (ConceptTaxonomy): The taxonomy to use.
            doc_text (str): Text of the document.
        """
        self.taxonomy = taxonomy
        self.doc_text = doc_text
        self.overall_concepts = None
        self.sentences_concepts = None
        self.aggregated_concepts = None
        self.hierarchy_distribution = None
        self.hierarchy_restriction = {}
        self.set_parameters()

    def set_parameters(
        self,
        top_k: int = 10,
        threshold: float = 0.3,
        weight_dev: Optional[Dict[int, float]] = None,
        kparm: float = 1.0,
    ) -> None:
        """Set the parameters for taxonomy concept assignment to document text.

        Args:
            top_k (int, optional): Number of concepts retained when checking similarity to text. Defaults to 10.
            threshold (float, optional): Minimum similarity for considering concepts. Defaults to 0.3.
            weight_dev (Dict[int, float], optional): Dictionary for setting sentence weights different from 1. Defaults to ``{0: 1.5}``.
            kparm (float, optional): Saturation parameter for aggregating similarities. Defaults to 1.0.
        """
        if weight_dev is None:
            weight_dev = {0: 1.5}
        self.top_k = top_k
        self.threshold = threshold
        self.weight_dev = weight_dev
        self.kparm = kparm

    def score_concepts_overall(self) -> None:
        """Finds concepts by simple similarity search for the whole text and stores them in attribute :pyattr:`overall_concepts`
        as a dict with concept key as key and similarity score as value.
        """
        self.overall_concepts = {
            x[0]: x[2]
            for x in self.taxonomy.get_concepts_in_text(self.doc_text, top_k=self.top_k, threshold=self.threshold)
        }

    def score_concepts_sentencewise(self) -> None:
        """Finds concepts by sentencewise similarity search and stores them in attribute :pyattr:`sentences_concepts`
        as a dict with concept key as key and a list of similarity scores for affected sentences, taking ``weight_dev`` into consideration.
        """
        language = _ISO_TO_NLTK_LANGUAGE.get(self.taxonomy.default_lang, "english")
        sentences = sent_tokenize(self.doc_text, language=language)
        weights = [1] * len(sentences)
        for i, g in self.weight_dev.items():
            if i < len(sentences):
                weights[i] = g
        concepts_scores = defaultdict(list)
        for w, s in zip(weights, sentences):
            for c in self.taxonomy.get_concepts_in_text(s, top_k=self.top_k, threshold=self.threshold):
                concepts_scores[c[0]].append(w * c[2])
        self.sentences_concepts = dict(concepts_scores)

    def aggregate_sentence_scores(self, min_agg_score: float = 0) -> None:
        """Identifies those of the concepts found by sentencewise similarity search which exceed a certain score after score aggregation
        and stores them in attribute :pyattr:`aggregated_concepts` as a dict with concept key as key and aggregated score as value.

        Args:
            min_agg_score (float, optional): Minimum score after sentence aggregation. Defaults to 0.
        """
        self.score_concepts_sentencewise()
        a_concepts = {}
        for c, v in self.sentences_concepts.items():
            a_score = 0
            for i, score in enumerate(sorted(v, reverse=True)):
                a_score += score * self.kparm / (i + self.kparm) / (i + 1 + self.kparm)
            if a_score >= min_agg_score:
                a_concepts[c] = a_score
        self.aggregated_concepts = a_concepts

    def find_best_paths(self, concept: str, n: int) -> List[List[str]]:
        """Finds paths from a given concept to the root of the taxonomy hierarchy along which the mean of aggregated concept scores is greatest.

        Args:
            concept (str): Key of concept.
            n (int): Number of paths to be returned.

        Returns:
            list: List of paths, each path being a list of concept keys.
        """
        scored_paths = [(p, sum([self.aggregated_concepts.get(ci, 0) for ci in p]) / len(p))
            for p in self.taxonomy.get_parent_concepts(concept)
        ]
        sorted_paths = sorted(scored_paths, key=lambda x: x[1], reverse=True)
        return [x[0] for x in sorted_paths[:n]]

    def evaluate_hierarchy(self, max_paths: int = 2) -> None:
        """Shares aggregated scores of concepts with their hierarchical predecessors and stores the results in attribute :pyattr:`hierarchy_distribution` as
        a dict of dicts, where the outer keys are hierarchy levels and the inner keys are concept keys, followed by the summed-up score shares.

        Args:
            max_paths (int, optional): Number of paths to the root considered per concept. Defaults to 2.
        """
        levels = defaultdict(lambda: defaultdict(float))
        for c, asc in self.aggregated_concepts.items():
            cps = self.find_best_paths(c, max_paths)
            for cp in cps:
                for i, ci in enumerate(reversed(cp[:-1])):
                    levels[i][ci] += asc / len(cps)
        self.hierarchy_distribution = {i: {c: v for c, v in cs.items()} for i, cs in levels.items()}

    def restrict_hierarchy(self, level: int, percentage: float) -> None:
        """Forms truncated lists of highest-scored concepts of a given hierarchy level such that the summed-up hierarchical scores of the remaining
        concepts exceed a given percentage of the total summed-up score, and stores the results in attribute :pyattr:`hierarchy_restriction` as a dict
        with the level number as key and the truncated list of concept keys as value.

        Args:
            level (int): Hierarchy level.
            percentage (float): Percentage of the total summed-up score that the remaining concepts must exceed.
        """
        if self.hierarchy_distribution.get(level):
            sum_list = []
            list_sum = 0
            for elem in sorted(self.hierarchy_distribution[level].items(), key=lambda x: x[1], reverse=True):
                sum_list.append((elem, list_sum))
                list_sum += elem[1]
            self.hierarchy_restriction[level] = [y[0][0] for y in sum_list if y[1] / list_sum * 100 < percentage]
        else:
            self.hierarchy_restriction[level] = []

    def get_relevant_concepts(
        self,
        min_score: Optional[float] = None,
        max_number: int = 100,
    ) -> List[Tuple[List[str], float]]:
        """Returns the hierarchical paths to all concepts exceeding a minimum aggregated score, as long as the steps of these paths are contained
        in the truncated level lists of concepts in ``hierarchy_restriction``.

        Args:
            min_score (float, optional): Minimum aggregated score. Defaults to ``None``, in which case the class attribute ``threshold`` is used.
            max_number (int, optional): Maximum number of concept paths returned. Defaults to 100.

        Returns:
            list: List of resulting concept paths, where each concept path is a pair of the list of concept keys constituting the path (root-first) and the aggregated score.
        """
        if min_score is None:
            min_score = self.threshold
        concept_paths = sorted(
            [(self.taxonomy.get_parent_concepts(c), s) for c, s in self.aggregated_concepts.items() if s >= min_score],
            key=lambda x: x[1], reverse=True
        )[:max_number]
        concept_paths = [(cp[-2::-1], cps[1]) for cps in concept_paths for cp in cps[0]]
        for level, rest in self.hierarchy_restriction.items():
            concept_paths = [cp for cp in concept_paths if set(cp[0][level:level + 1]).issubset(set(rest))]
        return concept_paths

    def identify_concepts(
        self,
        top_k: int,
        min_similarity: float,
        title_weight: float,
        saturation: float,
        min_agg: float,
        top_percent: float,
        second_percent: float,
        max_baseconcepts: int,
        max_paths: int,
        with_scores: bool = False,
    ) -> List[List[Union[Tuple[str, str], Tuple[str, str, float]]]]:
        """Convenience function for identifying concepts, given the various parameters of the algorithm.

        Args:
            top_k (int): Number of concepts retained when checking similarity to text.
            min_similarity (float): Minimum value for similarity on sentence level.
            title_weight (float): Weight for first sentence (which is supposed to be the title of the document).
            saturation (float): Saturation parameter for score aggregation.
            min_agg (float): Minimum value for aggregated score.
            top_percent (float): Percentage for hierarchy truncation on top level.
            second_percent (float): Percentage for hierarchy truncation on second level.
            max_baseconcepts (int): Maximum number of returned (leaf) concepts.
            max_paths (int): Number of paths to the root considered per concept.
            with_scores (bool, optional): Whether to include scores. Defaults to ``False``.

        Returns:
            list: List of lists of ``(concept name, concept key)`` pairs, or ``(concept name, concept key, aggregated score)`` triples when ``with_scores`` is ``True``.
        """
        self.set_parameters(top_k=top_k, threshold=min_similarity, weight_dev={0: title_weight}, kparm=saturation)
        self.aggregate_sentence_scores(min_agg_score=min_agg)
        self.evaluate_hierarchy(max_paths=max_paths)
        self.restrict_hierarchy(0, top_percent)
        self.restrict_hierarchy(1, second_percent)
        relevant_concepts_and_scores = self.get_relevant_concepts(min_score=min_agg, max_number=max_baseconcepts)
        relevant_concepts = [rc[0] for rc in relevant_concepts_and_scores]
        relevant_scores = [rc[1] for rc in relevant_concepts_and_scores]
        if not with_scores:
            return self.taxonomy.show_concepts(relevant_concepts)
        else:
            return self.taxonomy.show_concepts(relevant_concepts, scores=relevant_scores)
