"""
app/pages/02_Scanner.py

Stock scanner page.
"""

from __future__ import annotations

import streamlit as st

result = st.session_state.get(
    "result",
)


st.title(
    "🔎 Stock Scanner"
)


st.caption(
    "Ranked swing trading opportunities"
)


if result is None:

    st.warning(
        "Run the pipeline first."
    )

    st.stop()


scanner_data = result.get_data(
    "scan_results",
)


if scanner_data.empty:

    st.info(
        "No scanner results available."
    )

    st.stop()


st.subheader(
    "Scanner Filters"
)


col1, col2 = st.columns(2)


with col1:

    signal_filter = st.multiselect(

        "Signal",

        options=sorted(

            scanner_data["Signal"]

            .dropna()

            .unique()

            .tolist()

        )
        if "Signal" in scanner_data.columns
        else [],

    )


with col2:

    top_n = st.slider(

        "Top Stocks",

        min_value=5,

        max_value=100,

        value=20,

    )


filtered = scanner_data.copy()


if signal_filter:

    filtered = filtered.loc[

        filtered["Signal"]

        .isin(signal_filter)

    ]


st.subheader(
    "Ranked Opportunities"
)


display_columns = [

    column

    for column in [

        "Symbol",

        "Signal",

        "Rank",

        "Score",

        "Close",

        "Target",

        "StopLoss",

    ]

    if column in filtered.columns

]


st.dataframe(

    filtered[display_columns]

    .head(top_n),

    use_container_width=True,

)


st.download_button(

    "Download Scanner CSV",

    filtered.to_csv(
        index=False,
    ),

    file_name="scanner_results.csv",

    mime="text/csv",

)