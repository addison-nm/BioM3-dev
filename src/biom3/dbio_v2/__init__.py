"""biom3.dbio_v2 — simplified, self-contained database parsers.

A clean-slate rewrite of dataset construction. Parsers live in
``biom3.dbio_v2.parsers``: one self-contained file per source database,
each exposing ``iter_records(path) -> Iterator[Record]``. Built alongside
the legacy ``biom3.dbio`` package; the legacy build pipeline is untouched
until builders are rewired in a later pass.
"""
