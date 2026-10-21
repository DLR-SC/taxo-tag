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

(See also the included Jupyter notebook)

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

Turns a document into a set of relevant concept paths.

- `set_parameters(top_k=10, threshold=0.3, weight_dev=None, kparm=1.0)`
- `score_concepts_overall()` / `score_concepts_sentencewise()`
- `aggregate_sentence_scores(min_agg_score=0)`
- `find_best_paths(concept, n)`
- `evaluate_hierarchy(max_paths=2)` / `restrict_hierarchy(level, percentage)`
- `get_relevant_concepts(min_score=None, max_number=100)`
- `identify_concepts(...)` — all of the above in one call

## Development

This project uses [uv](https://docs.astral.sh/uv/). After cloning:

```bash
# Create the virtual environment and install with dev extras
uv sync --extra dev

# Build the distribution
uv build

# Upload to PyPI (after an `uv login` / twine login)
uv tool run twine upload dist/*
```

## License

MIT — DLR
