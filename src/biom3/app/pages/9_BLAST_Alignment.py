import streamlit as st

from biom3.viz.alignment import blast_sequence, superimpose
from biom3.viz.viewer import view_pdb, view_overlay
from biom3.viz.sources import (
    fetch_rcsb_pdb,
    fetch_alphafold,
    parse_pdb_id,
    parse_uniprot_id,
    pdb_to_sequence,
)
from biom3.app._helpers import render_view, pick_pdb

st.header("BLAST + Structure Alignment")
st.write(
    "Supply a query protein (sequence or PDB), run BLAST, then for each hit "
    "fetch or fold a structure and align it to the query. The overlay and RMSD "
    "are shown per-hit."
)


def _obtain_query(q_source: str):
    """Return (query_pdb_str or None, query_seq or None)."""
    query_pdb = None
    query_seq = None

    if q_source == "Browse / Upload PDB":
        query_pdb = pick_pdb(key="ba_query_pdb")
        if query_pdb:
            query_seq = pdb_to_sequence(query_pdb)
            if query_seq:
                st.caption(f"Extracted {len(query_seq)}-aa sequence from PDB")
            else:
                st.warning("Could not extract a sequence from the PDB.")
    elif q_source == "Paste PDB text":
        raw = st.text_area("Paste PDB content", height=200, key="ba_paste_pdb")
        if raw.strip():
            query_pdb = raw
            query_seq = pdb_to_sequence(query_pdb)
    elif q_source == "Sequence (will be folded)":
        raw = st.text_input(
            "Amino acid sequence (single-letter codes)", key="ba_paste_seq",
        )
        if raw.strip():
            query_seq = raw.strip().replace("\n", "").replace(" ", "")
            cached = st.session_state.get("ba_folded_query")
            if cached and cached.get("seq") == query_seq:
                query_pdb = cached["pdb"]
            if st.button("Fold query with ESMFold", key="ba_fold_query"):
                with st.spinner("Folding query..."):
                    try:
                        from biom3.viz.folding import fold_sequence
                        query_pdb = fold_sequence(query_seq)
                        st.session_state["ba_folded_query"] = {
                            "seq": query_seq, "pdb": query_pdb,
                        }
                    except ImportError:
                        st.error(
                            "ESMFold not available. Install: "
                            "`pip install 'fair-esm[esmfold]' omegaconf`"
                        )
                    except Exception as e:
                        st.error(f"Folding failed: {e}")
    return query_pdb, query_seq


def _get_hit_structure(h, source_choice: str, pdb_id: str | None, uniprot_id: str | None):
    if source_choice.startswith("Fetch RCSB"):
        return fetch_rcsb_pdb(pdb_id)
    if source_choice.startswith("Fetch AlphaFold"):
        return fetch_alphafold(uniprot_id)
    from biom3.viz.folding import fold_sequence
    return fold_sequence(h.hit_seq.replace("-", ""))


def _source_options(pdb_id: str | None, uniprot_id: str | None) -> list[str]:
    opts = []
    if pdb_id:
        opts.append(f"Fetch RCSB ({pdb_id})")
    if uniprot_id:
        opts.append(f"Fetch AlphaFoldDB ({uniprot_id})")
    opts.append("Fold with ESMFold (hit sequence)")
    return opts


def _run_page():
    q_source = st.radio(
        "Query source",
        ["Browse / Upload PDB", "Paste PDB text", "Sequence (will be folded)"],
        horizontal=True,
    )
    query_pdb, query_seq = _obtain_query(q_source)

    if query_pdb:
        st.subheader("Query structure")
        render_view(view_pdb(query_pdb), height=320)

    if not query_seq:
        st.info("Provide a query to continue.")
        return

    st.subheader("BLAST parameters")
    c1, c2, c3 = st.columns(3)
    with c1:
        database = st.selectbox(
            "Database", ["pdb", "swissprot", "nr", "refseq_protein"], key="ba_db",
        )
    with c2:
        max_hits = st.number_input("Max hits", 1, 30, 5, key="ba_max_hits")
    with c3:
        e_thresh = st.number_input(
            "E-value threshold", 0.0001, 100.0, 10.0, format="%.4f", key="ba_e",
        )

    if st.button("Run BLAST", type="primary", key="ba_run_blast"):
        if not query_pdb:
            st.error("Fold or upload a query structure first; alignment needs one.")
        else:
            with st.spinner("Running BLAST (30-60s)..."):
                try:
                    hits = blast_sequence(
                        query_seq,
                        database=database,
                        max_hits=int(max_hits),
                        e_value=e_thresh,
                    )
                except Exception as e:
                    st.error(f"BLAST failed: {e}")
                    return
            st.session_state["ba_hits"] = hits
            st.session_state["ba_query_pdb_for_hits"] = query_pdb
            st.session_state["ba_hit_structures"] = {}
            st.session_state["ba_alignments"] = {}

    hits = st.session_state.get("ba_hits", [])
    query_for_hits = st.session_state.get("ba_query_pdb_for_hits")
    if not hits or not query_for_hits:
        return

    st.subheader(f"{len(hits)} hits — obtain structure + align to query")

    hit_structures = st.session_state.setdefault("ba_hit_structures", {})
    alignments = st.session_state.setdefault("ba_alignments", {})

    for i, h in enumerate(hits):
        pdb_id = parse_pdb_id(h.hit_id, h.hit_def)
        uniprot_id = parse_uniprot_id(h.hit_id, h.hit_def)

        header = f"{h.hit_id} — {h.percent_identity:.1f}% identity, E={h.e_value:.1e}"
        with st.expander(header, expanded=(i == 0)):
            st.write(f"**{h.hit_def}**")
            if pdb_id or uniprot_id:
                tags = []
                if pdb_id:
                    tags.append(f"PDB: `{pdb_id}`")
                if uniprot_id:
                    tags.append(f"UniProt: `{uniprot_id}`")
                st.caption(" · ".join(tags))

            options = _source_options(pdb_id, uniprot_id)
            source_choice = st.selectbox(
                "Structure source",
                options,
                key=f"ba_src_{i}",
            )

            if st.button("Get structure + align", key=f"ba_go_{i}"):
                try:
                    with st.spinner("Fetching/folding structure..."):
                        pdb_str = _get_hit_structure(h, source_choice, pdb_id, uniprot_id)
                    hit_structures[h.hit_id] = pdb_str
                    with st.spinner("Superimposing on query..."):
                        result = superimpose(query_for_hits, pdb_str)
                    alignments[h.hit_id] = result
                except ImportError:
                    st.error(
                        "ESMFold not available. Install: "
                        "`pip install 'fair-esm[esmfold]' omegaconf`"
                    )
                except Exception as e:
                    st.error(f"Failed: {e}")

            result = alignments.get(h.hit_id)
            if result is not None:
                m1, m2 = st.columns(2)
                m1.metric("RMSD", f"{result.rmsd:.3f} Å")
                m2.metric("Paired CA atoms", result.n_atoms)
                v = view_overlay(
                    [result.fixed_pdb, result.moving_pdb],
                    labels=["Query", h.hit_id],
                    colors=["#4080ff", "#ff7040"],
                )
                render_view(v, height=400)


_run_page()
