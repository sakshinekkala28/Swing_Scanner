"""
app/pages/01_Dashboard.py

Main dashboard page.
"""

from __future__ import annotations

import streamlit as st

pipeline = st.session_state.get("pipeline")
result = st.session_state.get("result")


st.title("📈 Swing Scanner Dashboard")

st.caption(
    "Institutional Swing Trading Analytics Platform"
)


st.subheader("Pipeline Status")

if pipeline is None:

    st.warning(
        "Pipeline not initialized."
    )

else:

    health = pipeline.health_check()

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Status",
            health["status"],
        )

    with c2:
        st.metric(
            "Completed",
            health["completed"],
        )

    with c3:
        st.metric(
            "Execution Seconds",
            health["execution_seconds"],
        )


st.subheader("Results Overview")

if result is None:

    st.info(
        "Run the pipeline to view results."
    )

else:

    data = result.data

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Market Rows",
            len(data.get("market_data", [])),
        )

    with c2:
        st.metric(
            "Indicators",
            len(data.get("indicator_data", [])),
        )

    with c3:
        st.metric(
            "Signals",
            len(data.get("scan_results", [])),
        )

    with c4:
        st.metric(
            "Trades",
            len(data.get("trade_log", [])),
        )


st.subheader("Top Scanner Results")

if result is not None:

    scanner_data = result.get_data(
        "scan_results",
    )

    if not scanner_data.empty:

        st.dataframe(
            scanner_data.head(20),
            use_container_width=True,
        )

    else:

        st.info(
            "No scanner results available."
        )