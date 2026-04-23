from dataclasses import asdict

import pandas as pd
import streamlit as st

from biom3.viz.alignment import blast_sequence
from biom3.viz.viewer import view_pdb
from biom3.app._helpers import render_view, render_colored_sequence

st.header("BLAST Sequence Search")
st.write("Run a remote BLAST search against NCBI databases.")

sequence = st.text_area("Protein sequence (single-letter amino acid codes)", height=100)

col1, col2, col3 = st.columns(3)
with col1:
    database = st.selectbox("Database", ["nr", "swissprot", "pdb", "refseq_protein"])
with col2:
    max_hits = st.number_input("Max hits", min_value=1, max_value=50, value=10)
with col3:
    e_thresh = st.number_input(
        "E-value threshold", min_value=0.0001, max_value=100.0, value=10.0, format="%.4f",
    )

if sequence.strip() and st.button("Run BLAST"):
    clean_seq = sequence.strip().replace("\n", "").replace(" ", "")
    with st.spinner("Running BLAST (this may take 30-60 seconds)..."):
        try:
            hits = blast_sequence(
                clean_seq, database=database, max_hits=max_hits, e_value=e_thresh,
            )
        except Exception as e:
            st.error(f"BLAST failed: {e}")
            hits = []
    st.session_state["blast_hits"] = hits
    st.session_state.pop("blast_folded", None)

hits = st.session_state.get("blast_hits", [])

_MATCH_COLOR = (160, 220, 140)
_POSITIVE_COLOR = (240, 200, 120)
_MISMATCH_COLOR = (240, 170, 160)
_GAP_COLOR = (200, 200, 200)


def _alignment_colors(midline: str, query: str, subject: str) -> list[tuple[int, int, int]]:
    colors = []
    for i, m in enumerate(midline):
        if m.isalpha():
            colors.append(_MATCH_COLOR)
        elif m == "+":
            colors.append(_POSITIVE_COLOR)
        elif (i < len(query) and query[i] == "-") or (i < len(subject) and subject[i] == "-"):
            colors.append(_GAP_COLOR)
        else:
            colors.append(_MISMATCH_COLOR)
    return colors


if hits:
    st.subheader(f"{len(hits)} hits found")

    df = pd.DataFrame([asdict(h) for h in hits])
    summary_cols = [
        "hit_id", "percent_identity", "e_value", "score",
        "align_length", "identities", "positives", "hit_def",
    ]
    df_summary = df[[c for c in summary_cols if c in df.columns]]
    st.dataframe(df_summary, use_container_width=True, hide_index=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download hits as CSV",
        data=csv_bytes,
        file_name="blast_hits.csv",
        mime="text/csv",
    )

    st.subheader("Per-hit alignments")
    folded = st.session_state.setdefault("blast_folded", {})

    for i, h in enumerate(hits):
        with st.expander(
            f"{h.hit_id} — {h.percent_identity:.1f}% identity, E={h.e_value:.1e}",
        ):
            st.write(f"**{h.hit_def}**")
            st.write(
                f"Score: {h.score:.0f} | Identities: {h.identities}/{h.align_length} "
                f"| Positives: {h.positives}/{h.align_length}",
            )

            colors = _alignment_colors(h.midline, h.query_seq, h.hit_seq)
            query_html = render_colored_sequence(
                h.query_seq, colors=colors, label="Query", show_positions=False,
            )
            midline_html = render_colored_sequence(
                h.midline, colors=None, label="", show_positions=False,
            )
            subject_html = render_colored_sequence(
                h.hit_seq, colors=colors, label="Sbjct", show_positions=False,
            )
            st.markdown(query_html, unsafe_allow_html=True)
            st.markdown(midline_html, unsafe_allow_html=True)
            st.markdown(subject_html, unsafe_allow_html=True)

            fold_key = f"fold_{i}_{h.hit_id}"
            if st.button(f"Fold this hit with ESMFold", key=fold_key):
                clean = h.hit_seq.replace("-", "")
                with st.spinner("Folding hit with ESMFold..."):
                    try:
                        from biom3.viz.folding import fold_sequence
                        folded[h.hit_id] = fold_sequence(clean)
                    except ImportError:
                        st.error(
                            "ESMFold not available. "
                            "Install with: `pip install 'fair-esm[esmfold]' omegaconf`"
                        )
                    except Exception as e:
                        st.error(f"Folding failed: {e}")

            pdb = folded.get(h.hit_id)
            if pdb:
                render_view(view_pdb(pdb), height=400)
elif sequence.strip() and "blast_hits" in st.session_state:
    st.info("No hits found.")
