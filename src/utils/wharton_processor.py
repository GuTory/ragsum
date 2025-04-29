'''
Helper functions for loading the already scraped dataframes.
`load_all_available_transcripts` loads and aggregates all local csv-s
'''

import os
from pathlib import Path
from typing import Optional, List
import pandas as pd

from utils import setup_logger

logger = setup_logger(__name__)


def get_root_path() -> Path:
    '''Helper function that returns the root path of the project.'''
    return Path(os.getcwd()).parent


def load_if_scraped(company_id: str) -> Optional[pd.DataFrame]:
    '''
    Transcript loader dataset based on company_id, if nothing found, returns None
    '''
    file_path = get_root_path() / 'data' / f'{company_id}.csv'
    if file_path.exists():
        df = pd.read_csv(
            file_path,
            sep='\t',
            quoting=1,
            escapechar='\\',
            doublequote=True,
            quotechar='"',
        )
        logger.info('Successfully loaded local transcripts for %s', company_id)
        return df
    logger.debug('No local transcripts found for %s', company_id)
    return None


def list_of_companies() -> List[str]:
    '''
    Loads the list of company IDs from the companies.csv file
    '''
    file_path = get_root_path() / 'data' / 'companies.csv'
    companies = pd.read_csv(file_path, sep='\t')
    return companies.companyid.astype(str).tolist()


def load_all_available_transcripts() -> pd.DataFrame:
    '''
    Loads and concatenates all available transcript CSVs
    that match company IDs in the company list.
    Returns a combined DataFrame.
    '''
    data_folder = get_root_path() / 'data'
    company_ids = set(list_of_companies())
    all_dfs = []

    for file in data_folder.glob('*.csv'):
        company_id = file.stem
        if company_id in company_ids:
            logger.info(company_id)
            df = load_if_scraped(company_id=company_id)
            if not df.empty:
                all_dfs.append(df)
                logger.info('Successfully loaded %s', file.name)

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        logger.info('Successfully combined all matching transcripts: %s', combined_df.shape)
        return combined_df

    logger.info('No matching transcript files found.')
    return pd.DataFrame()


if __name__ == '__main__':
    logger.info('Loading all available transcripts...')
    all_transcripts_df = load_all_available_transcripts()

    if not all_transcripts_df.empty:
        logger.info('Total transcripts loaded: %s', len(all_transcripts_df))
    else:
        logger.info('No transcripts were loaded.')
