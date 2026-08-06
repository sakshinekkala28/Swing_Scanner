"""
app/pages/03_Backtest.py

Backtest analysis page.
"""

from __future__ import annotations

import streamlit as st

result = st.session_state.get(
    "result",
)


st.title(
    "📊 Backtest Analysis"
)


st.caption(
    "Strategy performance and historical validation"
)


if result is None:

    st.warning(
        "Run the pipeline first."
    )

    st.stop()


trade_log = result.get_data(
    "trade_log",
)

equity_curve = result.get_data(
    "equity_curve",
)

statistics = result.statistics


st.subheader(
    "Performance Summary"
)


if statistics:

    cols = st.columns(4)

    metrics = list(
        statistics.items()
    )[:4]

    for col, (key, value) in zip(
        cols,
        metrics,
    ):

        with col:

            st.metric(
                key.replace("_", " ").title(),
                value,
            )

else:

    st.info(
        "No statistics available."
    )


st.subheader(
    "Equity Curve"
)


if not equity_curve.empty:

    if "Equity" in equity_curve.columns:

        st.line_chart(
            equity_curve.set_index(
                equity_curve.columns[0]
            )["Equity"]
        )

    else:

        st.dataframe(
            equity_curve,
            use_container_width=True,
        )

else:

    st.info(
        "No equity curve available."
    )


st.subheader(
    "Trade Log"
)


if not trade_log.empty:

    st.dataframe(

        trade_log,

        use_container_width=True,

    )

    st.download_button(

        "Download Trade Log",

        trade_log.to_csv(
            index=False,
        ),

        file_name="trade_log.csv",

        mime="text/csv",

    )

else:

    st.info(
        "No trades available."
    )