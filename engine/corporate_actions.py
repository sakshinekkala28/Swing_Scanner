"""
Corporate actions engine.
"""


class CorporateActionsEngine:

    def fetch_latest(self):

        import pandas as pd

        return pd.DataFrame(
            columns=[
                "Symbol",
                "BONUS",
                "SPLIT",
                "DIVIDEND",
                "RIGHTS",
            ]
        )