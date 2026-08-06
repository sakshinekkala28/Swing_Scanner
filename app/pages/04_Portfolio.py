"""
app/pages/04_Portfolio.py

Portfolio monitoring page.
"""

from __future__ import annotations

import streamlit as st

result = st.session_state.get(
    "result",
)


st.title(
    "💼 Portfolio Dashboard"
)


st.caption(
    "Portfolio allocation and risk overview"
)


if result is None:

    st.warning(
        "Run the pipeline first."
    )

    st.stop()


data = result.data

scan_results = result.get_data(
    "scan_results",
)


trade_log = result.get_data(
    "trade_log",
)


st.subheader(
    "Portfolio Summary"
)


c1, c2, c3 = st.columns(3)


with c1:

    st.metric(
        "Positions",
        len(scan_results),
    )


with c2:

    st.metric(
        "Trades",
        len(trade_log),
    )


with c3:

    st.metric(
        "Datasets",
        len(data),
    )


st.subheader(
    "Current Holdings"
)


if not scan_results.empty:

    columns = [

        col

        for col in [

            "Symbol",

            "Signal",

            "Rank",

            "Score",

            "Close",

        ]

        if col in scan_results.columns

    ]


    st.dataframe(

        scan_results[columns],

        use_container_width=True,

    )


else:

    st.info(
        "No portfolio data available."
    )


st.subheader(
    "Trade Exposure"
)


if not trade_log.empty:

    st.dataframe(

        trade_log,

        use_container_width=True,

    )

else:

    st.info(
        "No trade exposure available."
    )