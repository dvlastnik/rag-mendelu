import traceback
import pandas as pd

from etl.BaseEtl import BaseEtl, ETLState
from database.base.Document import Document
from utils.logging_config import get_logger
from text_embedding_api.TextEmbeddingService import TextEmbeddingService

logger = get_logger(__name__)

class Mzdr1DataEtl(BaseEtl):
    new_headers = [
        "indicator_label","ekonomic_sector", "year",
        "state","region", "province", "value"
    ]

    def _row_to_document(self, row) -> Document:

        if row['province'] == row['region'] == row['state']:
            text = (
                f"V roce {row['year']} byl ukazatel '{row['indicator_label']}' "
                f"v sektoru '{row['ekonomic_sector']}' "
                f"v oblasti {row['state']} "
                f"ve výši {row['value']}."
            )
        else:
            text = (
                f"V roce {row['year']} byl ukazatel '{row['indicator_label']}' "
                f"v sektoru '{row['ekonomic_sector']}' "
                f"v oblasti {row['province']} {row['region']} {row['state']} "
                f"ve výši {row['value']}."
            )

        return Document(id=None, text=text, embedding=None, metadata=row.to_dict())


    def transform(self) -> None:
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
                "Ukazatel": self.new_headers[0],
                "Odvětví ekonomické činnosti": self.new_headers[1],
                "Roky": self.new_headers[2],
                "ČR, regiony, kraje-Stát": self.new_headers[3],
                "ČR, regiony, kraje-Region": self.new_headers[4],
                "ČR, regiony, kraje-Kraj": self.new_headers[5],
                "Hodnota": self.new_headers[6]
            }, inplace=True)

            df["year"] = df["year"].astype(int)
            df["value"] = pd.to_numeric(df["value"], errors="coerce").round(1)
            df["region"] = df["region"].fillna("Česko")
            df["province"] = df["province"].fillna("Česko")

            for col in self.new_headers:
                df[col] = df[col].astype(str).str.strip()
            
            self.df = df

            logger.info(f"Transformed data to shape: {self.df.shape}")
            self.state = ETLState.TRANSFORMED
        except Exception as e:
            logger.exception("Error during transform step")
            traceback.print_exc()
            self.state = ETLState.FAILED
            return

            

