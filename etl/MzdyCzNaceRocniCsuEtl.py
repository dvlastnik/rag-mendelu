from etl.BaseEtl import BaseEtl
import pandas as pd

class MzdyCzNaceRocniCsuEtl(BaseEtl):
    def transform(self) -> None:
        # renaming columns
        df = self.df[[
            "Ukazatel",
            "IndicatorType",
            "CZNACEMZDY",
            "Odvětví ekonomické činnosti",
            "Roky",
            "Hodnota",
            "ČR, regiony, kraje-Stát",
            "ČR, regiony, kraje-Region",
            "ČR, regiony, kraje-Kraj"
        ]].copy()

        df.rename(columns={
            "Ukazatel": "indicator_label",
            "IndicatorType": "indicator_code",
            "CZNACEMZDY": "nace_code",
            "Odvětví ekonomické činnosti": "nace_description",
            "Roky": "year",
            "Hodnota": "value_czk",
            "ČR, regiony, kraje-Stát": "state",
            "ČR, regiony, kraje-Region": "region",
            "ČR, regiony, kraje-Kraj": "kraj"
        }, inplace=True)

        df["year"] = df["year"].astype(int)
        df["value_czk"] = pd.to_numeric(df["value_czk"], errors="coerce")

        df.dropna(subset=["value_czk"], inplace=True)

        for col in ["indicator_label", "indicator_code", "nace_code", "nace_description", "state", "region", "kraj"]:
            df[col] = df[col].astype(str).str.strip()

        self.df = df
        