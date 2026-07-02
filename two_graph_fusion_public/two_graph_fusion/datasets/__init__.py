"""Dataset-specific loaders.

Currently only TwiBot-22 is supported; future modules (``twibot20.py``,
``cresci2017.py``, ``wiki_socks.py``) will follow the same interface
sketched in ``twibot22.py``:

- ``discover_<dataset>(root) -> Paths``
- ``load_labels_and_splits(paths) -> DataFrame``
- ``stream_timestamps_for_users(...) -> dict[user_id, list[float]]``
"""

from two_graph_fusion.datasets import twibot22

__all__ = ["twibot22"]
