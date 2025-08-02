import traceback
from etl.BaseEtl import BaseEtl, ETLState
import pandas as pd

from utils.logging_config import get_logger

logger = get_logger(__name__)

class Mzdr1DataEtl(BaseEtl):
    def transform(self) -> None:
        new_headers = [
            "indicator_label","ekonomic_sector", "year",
            "state","region", "province", "value"
        ]

        try:
            # use only the data that we want
            df = self.df[[
                "Ukazatel",
                "Odvětví ekonomické činnosti",
                "Roky",
                "ČR, regiony, kraje-Stát",
                "ČR, regiony, kraje-Region",
                "ČR, regiony, kraje-Kraj",
                "Hodnota"
            ]].copy()

            df.rename(columns={
                "Ukazatel": new_headers[0],
                "Odvětví ekonomické činnosti": new_headers[1],
                "Roky": new_headers[2],
                "ČR, regiony, kraje-Stát": new_headers[3],
                "ČR, regiony, kraje-Region": new_headers[4],
                "ČR, regiony, kraje-Kraj": new_headers[5],
                "Hodnota": new_headers[6]
            }, inplace=True)

            df["year"] = df["year"].astype(int)
            df["value"] = pd.to_numeric(df["value"], errors="coerce").round(1)
            df["region"] = df["region"].fillna("Česko")
            df["province"] = df["province"].fillna("Česko")

            for col in new_headers:
                df[col] = df[col].astype(str).str.strip()
            
            self.df = df

            logger.info(f"Transformed data to shape: {self.df.shape}")
            self.state = ETLState.TRANSFORMED
        except Exception as e:
            logger.exception("Error during transform step")
            traceback.print_exc()
            self.state = ETLState.FAILED
            return

            

