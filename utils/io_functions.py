import logging
import pandas as pd
from typing import Optional
from pathlib import Path
import subprocess


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

def run(args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        args.split(),
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        text=True
    )

def popen(args) -> subprocess.Popen:
    return subprocess.Popen(
        args.split(),
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        text=True
    )