# New figure PNG contact sheet archive

This directory contains PNG renders of every tracked `paper_results` PDF whose
exact SHA256 is absent from `/u/yucheny8/nautilus-mlsys27/figures`. It includes
new operator plots, FP8 plots, e2e graphs, historical diagnostic figures, and
newly archived reproduction outputs. PDFs copied byte-for-byte from the old
paper figure directory are excluded.

`manifest.json` maps every PNG back to its PDF, records page counts and hashes,
and records the source branch revision. Renders use Poppler at 144 dpi.
