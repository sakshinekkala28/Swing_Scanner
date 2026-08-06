"""
app/pages/06_Settings.py

Application settings and system status page.
"""

from __future__ import annotations

import streamlit as st

from config import settings

pipeline = st.session_state.get(
    "pipeline",
)

result = st.session_state.get(
    "result",
)


st.title(
    "⚙️ Settings"
)


st.caption(
    "Application configuration and system health"
)


###############################################################################
# Application Information
###############################################################################

st.subheader(
    "Application"
)


c1, c2 = st.columns(2)


with c1:

    st.metric(
        "Name",
        "Swing Scanner",
    )


with c2:

    st.metric(
        "Environment",
        getattr(
            settings.app,
            "environment",
            "Unknown",
        ),
    )


###############################################################################
# Pipeline Health
###############################################################################

st.subheader(
    "Pipeline Health"
)


if pipeline is None:

    st.warning(
        "Pipeline not initialized."
    )

else:

    health = pipeline.health_check()

    st.json(
        health,
    )


###############################################################################
# Pipeline Metadata
###############################################################################

st.subheader(
    "Execution Metadata"
)


if result is None:

    st.info(
        "No pipeline execution available."
    )

else:

    st.json(
        result.metadata,
    )


###############################################################################
# Configuration
###############################################################################

st.subheader(
    "Configuration"
)


st.json(

    {

        "version":

            getattr(

                settings.app,

                "version",

                "Unknown",

            ),

        "timezone":

            getattr(

                settings.app,

                "timezone",

                "Unknown",

            ),

    }

)