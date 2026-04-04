"""BioM3 Web App — Streamlit multi-page application.

Launch with:
    streamlit run src/biom3/app/__init__.py

Or via entry point:
    biom3_app
"""

import streamlit as st


def main():
    import argparse, os, subprocess, sys
    from pathlib import Path

    from biom3.app.config import SETTINGS_ENV_VAR

    parser = argparse.ArgumentParser(description="BioM3 Streamlit app")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to app settings JSON (overrides bundled defaults)",
    )
    args, extra = parser.parse_known_args()
    if args.config:
        os.environ[SETTINGS_ENV_VAR] = str(Path(args.config).resolve())
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", __file__] + extra,
        check=True,
    )


st.set_page_config(
    page_title="BioM3",
    page_icon=":dna:",
    layout="wide",
)

st.title("BioM3")
st.markdown(
    "Multi-stage framework for generating novel protein sequences "
    "guided by natural language prompts."
)
st.markdown("Select a page from the sidebar to get started.")

st.sidebar.success("Select a page above.")
