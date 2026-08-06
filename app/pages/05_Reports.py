"""
app/pages/05_Reports.py

Reports and exports page.
"""

from __future__ import annotations

import streamlit as st

result = st.session_state.get(
    "result",
)


st.title(
    "📄 Reports"
)


st.caption(
    "Generated analytics reports and exports"
)


if result is None:

    st.warning(
        "Run the pipeline first."
    )

    st.stop()


reports = result.reports


st.subheader(
    "Available Reports"
)


if not reports:

    st.info(
        "No reports available."
    )

    st.stop()


for name, dataframe in reports.items():

    st.markdown(
        f"### {name.replace('_', ' ').title()}"
    )


    if dataframe.empty:

        st.warning(
            "Report is empty."
        )

        continue


    st.dataframe(

        dataframe,

        use_container_width=True,

    )


    st.download_button(

        label=f"Download {name}.csv",

        data=dataframe.to_csv(
            index=False,
        ),

        file_name=f"{name}.csv",

        mime="text/csv",

    )


st.subheader(
    "Report Summary"
)


summary = result.summary()


st.json(
    summary,
)