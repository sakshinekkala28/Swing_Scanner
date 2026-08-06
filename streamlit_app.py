"""
streamlit_app.py

Main Streamlit entry point for Swing Scanner.
"""

from __future__ import annotations

import logging
from datetime import datetime

import streamlit as st

from workflows.market_pipeline import MarketPipeline

logger = logging.getLogger(__name__)


###############################################################################
# Page Configuration
###############################################################################

st.set_page_config(
    page_title="Swing Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


###############################################################################
# Session Initialization
###############################################################################

def initialize_session() -> None:
    """
    Initialize Streamlit session state.
    """

    if "pipeline" not in st.session_state:

        st.session_state.pipeline = MarketPipeline()


    if "result" not in st.session_state:

        st.session_state.result = None


    if "last_run" not in st.session_state:

        st.session_state.last_run = None


    if "error" not in st.session_state:

        st.session_state.error = None



###############################################################################
# Pipeline Execution
###############################################################################

def run_pipeline() -> None:
    """
    Execute market pipeline.
    """

    try:

        progress_bar = st.progress(
            0
        )

        status = st.empty()


        def progress_callback(
            message,
            value,
        ):

            status.info(
                message
            )

            progress_bar.progress(
                value
            )


        st.session_state.pipeline.progress_callback = (
            progress_callback
        )


        with st.spinner(
            "Executing pipeline..."
        ):

            result = st.session_state.pipeline.run()

            st.session_state.result = result

            st.session_state.last_run = datetime.utcnow()

            st.session_state.error = None


    except Exception as exc:

        logger.exception(
            "Pipeline execution failed."
        )

        st.session_state.error = str(exc)



###############################################################################
# Sidebar
###############################################################################

def render_sidebar() -> None:
    """
    Render application sidebar.
    """

    st.sidebar.title(
        "Swing Scanner"
    )


    if st.sidebar.button(
        "▶ Run Pipeline",
        use_container_width=True,
    ):

        run_pipeline()


    if st.sidebar.button(
        "🔄 Reset",
        use_container_width=True,
    ):

        st.session_state.result = None

        st.session_state.error = None



    st.sidebar.divider()


    if st.session_state.last_run:

        st.sidebar.info(

            f"Last Run:\n"
            f"{st.session_state.last_run}"

        )


    if st.session_state.error:

        st.sidebar.error(

            st.session_state.error

        )



###############################################################################
# Application
###############################################################################

def main() -> None:
    """
    Main application runner.
    """

    initialize_session()

    render_sidebar()


    st.title(
        "📈 Swing Scanner Platform"
    )


    st.caption(
        "Institutional Trading Research & Analytics Platform"
    )


    result = st.session_state.result


    if result is None:

        st.info(
            "Click 'Run Pipeline' to start analysis."
        )

    else:

        st.success(
            "Pipeline completed successfully."
        )

        st.json(
            result.summary()
        )


if __name__ == "__main__":

    main()