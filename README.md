# taxo-tag

Tag documents with terms from a concept hierarchy (taxonomy) using sentence
embeddings.

`taxo` builds a hierarchy of concepts, embeds their descriptions in a shared
vector space, and then finds the concepts most related to a document. It can
also aggregate sentence-level evidence along hierarchical paths to produce a
consistent set of top-level tags.

For the main ideas used in `taxo`, see
```
@inproceedings{dlr190966,
           title = {Strategy Comparison for Semantic Zero-Shot Taxonomy Filters},
           month = {October},
       booktitle = {4th International Open Search Symposium (OSSYM 2022)},
          author = {Hamm, Andreas},
            year = {2022},
             url = {https://elib.dlr.de/190966/}
}
```

## Installation

```bash
pip install taxo-tag
```

This is a uv-style package, so `uv` works too:

```bash
uv add taxo-tag
```

## Usage
Note that while the package is called taxo-tag on PyPi, import it simply as `taxo`.

Given a taxonomy (by this we mean a hierarchy of concepts, ordered in the structure of a directed acyclic graph (DAG)) - presented in tabular form as described below - and the text of a document, this package can be used to tag the document with the best-fitting taxonomy concepts.

The table used to specify the taxonomy is expected to be a Pandas dataframe. Thus, it can be imported from several tabular sources like csv files or Excel sheets. An example is the file `media.xlsx` included in this repository.

The table has one row for every node (concept) of the taxonomy. Formally, there is one special node needed, the so called root or top node, which is the ancestor of all other nodes in the hierarchy. (See the node with key `M00000000` in the media example.) 

Each row of the table has to specify:
- the concept's unique `key`
- language dependent labels for the concept
- short language dependent descriptions of the concept
- the keys of the hierarchical `parents` of the concept 

The parents' keys have to be given as a string that evaluates to a list in Python (e.g., `"['concept1', 'concept2']"`). In the case of a hierarchy that is a tree, every concept has of course only one parent.

Given a 2-letter ISO code `xy` for a language, the row headers for labels and descriptions are `label_xy` and `description_xy`, respectively. The table is expected to have English entries (`en`) as default, and more entries for the other languages needed. The code in this package is also meant to handle German (`de`); other languages need small obvious modifications.

Tagging is based on semantic similarity considerations as calculated by a Sentence Transformer model. Any model provided by the sentence_transformers package can be used. The small model `all-MiniLM-L6-v2` works well for English (but not so well for other languages).

The code fragment below shows the principles of usage, finally executing the tagging by calling the method `identify_concepts` which has several tuning parameters which can be adapted to specific constellations and use cases. 

```python
import networkx as nx
import pandas as pd
from sentence_transformers import SentenceTransformer

import taxo

# Build a taxonomy from a dataframe (one row per concept).
df = pd.read_excel("media.xlsx")
df.fillna("", inplace=True)

tax = taxo.ConceptTaxonomy(
    nx.DiGraph(),
    root="M00000000",          # key of the formal top concept
    transformer="all-MiniLM-L6-v2",
    default_lang="en",
)
tax.from_df(df)

# Tag a document.
doc = taxo.TaxedDocument(tax, text)
result = doc.identify_concepts(
    top_k=10,
    min_similarity=0.25,
    title_weight=2,
    saturation=1.5,
    min_agg=0.25,
    top_percent=80,
    second_percent=80,
    max_baseconcepts=7,
    max_paths=2,
    with_scores=True,
)
```

Each entry of `result` is a hierarchical path from the root to a concept, as a
list of `(name, key)` pairs (optionally with an aggregated score appended).

See also the included Jupyter notebook `usage.ipynb`.

## API

### `ConceptTaxonomy`

Holds the concept hierarchy and its embeddings.

- `__init__(taxgraph, root, long_descr=True, transformer=None, default_lang="en", classifier=None, token=None)`
- `from_df(df_concepts, rename_columns=None)` — add/update concepts from a dataframe
- `to_df()` — export the hierarchy to a dataframe
- `describe_concepts(lang=None, intro=None, long=True)` — prepare comparison sentences
- `embed_concepts(model=None)` — compute the concept embeddings
- `get_concepts_in_text(text, top_k=10, threshold=0.3, sim_func=None)` — similarity or zero-shot search
- `get_parent_concepts(child, top_parent=None, lang=None)` — paths from a concept to the root
- `show_concepts(concepts, lang=None, scores=None)` — format results for output

### `TaxedDocument`

Prepares a document for concept identification.

- `set_parameters(top_k=10, threshold=0.3, weight_dev=None, kparm=1.0)`
- `score_concepts_overall()` / `score_concepts_sentencewise()`
- `aggregate_sentence_scores(min_agg_score=0)`
- `find_best_paths(concept, n)`
- `evaluate_hierarchy(max_paths=2)` / `restrict_hierarchy(level, percentage)`
- `get_relevant_concepts(min_score=None, max_number=100)`
- `identify_concepts(...)` — all of the above in one call


## License

MIT — DLR. See LICENSE file.
