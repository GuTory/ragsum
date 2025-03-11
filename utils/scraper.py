import os
import logging
import uuid
import wrds
import pandas as pd
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from nltk.tokenize import word_tokenize


@dataclass
class WhartonCompanyIdSearchCache:
    id: int
    name: str
    df: pd.DataFrame
    transcripts: Optional[pd.DataFrame]


class WhartonScraper:
    """
    Wrapper Class for Scraping Wharton Transcripts database for a single company
    Based on Company name or id.
    """

    def __init__(
        self,
        connection: wrds.Connection = None,
    ):
        self.connection: wrds.Connection = (
            connection if connection is not None else wrds.Connection()
        )
        self._company_search_cache: WhartonCompanyIdSearchCache = None

    def __repr__(self):
        return f"WhartonScraper(id={self._company_search_cache.id}, name={self._company_search_cache.name})"

    def __str__(self):
        return f"WhartonScraper for company ({self._company_search_cache.name})"

    def pipeline(
        self, company_id: str, start: int = 2020, end: int = datetime.now().year
    ) -> None:
        """Full Pipeline for transcript acquisition from Wharton database, based on `companyid`

        Args:
            company_id (str): `companyid` to filter by
        """
        self.__get_company_by_id(company_id)
        self.__get_company_transcripts(start, end)
        self.__transcripts_to_csv()

    def __get_company_by_id(self, company_id: str) -> Optional[pd.DataFrame]:
        """
        Reaching out to Wharton database to see if `companyid` is present
        """
        if self._company_search_cache and self._company_search_cache.id == company_id:
            logging.debug(f"using cache on company: {company_id}")
            return self._company_search_cache.df

        if self.connection is None:
            return None

        select_company = f"""
            SELECT DISTINCT d.companyid, d.companyname
            FROM ciq.wrds_transcript_detail as d
            WHERE d.companyid = {company_id}
        """
        df: pd.DataFrame = self.connection.raw_sql(select_company)

        if df.shape[0] == 0:
            logging.debug(f"no results for company search")
            self._company_search_cache = None
            return None

        if df.shape[0] > 1:
            logging.debug(f"too many results for search: {df.shape[0]}")
            self._company_search_cache = None
            return None

        self._company_search_cache = WhartonCompanyIdSearchCache(
            id=company_id, name=df.companyname[0], df=df, transcripts=None
        )
        logging.info(f"information acquired for company: {company_id}")
        return df

    def __get_company_transcripts(self, start: int, end: int) -> Optional[pd.DataFrame]:
        """
        Acquiring company transcripts based on the cached `companyid`
        """
        if not self._company_search_cache:
            logging.debug("no company cache")
            return None
        if self._company_search_cache.transcripts:
            logging.debug("transcripts already cached")
            return self._company_search_cache.transcripts

        if self.connection is None:
            return None

        logging.info(f"scraping transcripts between {start} and {end}")
        query = f"""
            SELECT a.*, b.*, c.componenttext
            FROM (
                  SELECT * 
                  FROM ciq.wrds_transcript_detail
                  WHERE companyid = {self._company_search_cache.id}
                    AND date_part('year', mostimportantdateutc) BETWEEN {start} AND {end}
                 ) AS a
            JOIN ciq.wrds_transcript_person AS b
              ON a.transcriptid = b.transcriptid
            JOIN ciq.ciqtranscriptcomponent AS c
              ON b.transcriptcomponentid = c.transcriptcomponentid
            ORDER BY a.transcriptid, b.componentorder;
            """
        df = self.connection.raw_sql(query)
        df = df.drop(["transcriptpersonname"], axis=1)
        transcripts: pd.DataFrame = (
            df.groupby(
                [
                    "companyid",
                    "companyname",
                    "mostimportantdateutc",
                    "mostimportanttimeutc",
                    "headline",
                ]
            )
            .apply(
                lambda group: "\n".join(
                    f"{row['speakertypename']}: {row['componenttext']}"
                    for _, row in group.iterrows()
                ),
                include_groups=False,
            )
            .reset_index(name="full_text")
        )
        transcripts.companyid = transcripts.companyid.astype(int)
        transcripts["uuid"] = uuid.uuid4()
        transcripts["word_count"] = transcripts["full_text"].apply(
            lambda x: len(str(x).split())
        )
        transcripts["word_count_nltk"] = transcripts["full_text"].apply(
            lambda x: len(word_tokenize(str(x)))
        )

        self._company_search_cache.transcripts = transcripts
        logging.info(
            f"transcripts acquired for company: {self._company_search_cache.id} with a shape: {transcripts.shape}"
        )
        return transcripts

    def __transcripts_to_csv(self) -> None:
        """
        Writing transcript dataset to file if it is present
        """
        if self._company_search_cache.transcripts is None:
            logging.debug("no transcript records.")
            return

        self._company_search_cache.transcripts.to_csv(
            Path("..") / "data" / f"{self._company_search_cache.id}.csv",
            sep="\t",
            index=False,
            quoting=1,
            escapechar="\\",
            doublequote=True,
            quotechar='"',
            lineterminator="\n",
        )
        logging.info(
            f"transcripts successfully written to {self._company_search_cache.id}.csv"
        )
