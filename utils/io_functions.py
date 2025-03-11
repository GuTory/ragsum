import logging
import pandas as pd
from typing import Optional
from pathlib import Path


def load_if_scraped(company_id: str) -> Optional[pd.DataFrame]:
    file_path = Path('..') / 'data' / f'{company_id}.csv'
    if file_path.exists():
        df = pd.read_csv(
            file_path,
            sep='\t',
            quoting=1,
            escapechar='\\',
            doublequote=True,
            quotechar='"',
        )
        logging.info('successfully loaded local transcripts')
        return df
    else:
        logging.debug('no local transcripts found')
    return None