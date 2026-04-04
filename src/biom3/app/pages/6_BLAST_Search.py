import streamlit as st

from biom3.viz.alignment import blast_sequence

st.header("BLAST Sequence Search")
st.write("Run a remote BLAST search against NCBI databases.")

sequence = st.text_area("Protein sequence (single-letter amino acid codes)", height=100)

col1, col2, col3 = st.columns(3)
with col1:
    database = st.selectbox("Database", ["nr", "swissprot", "pdb", "refseq_protein"])
with col2:
    max_hits = st.number_input("Max hits", min_value=1, max_value=50, value=10)
with col3:
    e_thresh = st.number_input("E-value threshold", min_value=0.0001, max_value=100.0, value=10.0, format="%.4f")

if sequence.strip() and st.button("Run BLAST"):
    clean_seq = sequence.strip().replace("\n", "").replace(" ", "")
    with st.spinner("Running BLAST (this may take 30-60 seconds)..."):
        try:
            hits = blast_sequence(clean_seq, database=database, max_hits=max_hits, e_value=e_thresh)
        except Exception as e:
            st.error(f"BLAST failed: {e}")
            hits = []

    if hits:
        st.subheader(f"{len(hits)} hits found")
        for i, h in enumerate(hits):
            with st.expander(f"{h.hit_id} — {h.percent_identity:.1f}% identity, E={h.e_value:.1e}"):
                st.write(f"**{h.hit_def}**")
                st.write(f"Score: {h.score:.0f} | Identities: {h.identities}/{h.align_length} "
                         f"| Positives: {h.positives}/{h.align_length}")
                st.code(
                    f"Query:  {h.query_seq}\n"
                    f"        {h.midline}\n"
                    f"Sbjct:  {h.hit_seq}",
                    language=None,
                )
    elif sequence.strip():
        st.info("No hits found.")
