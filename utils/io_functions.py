'''
Module for using shell and file system (mostly csv files).
'''

import logging
from typing import Optional
from pathlib import Path
import subprocess
import pandas as pd


def load_if_scraped(company_id: str) -> Optional[pd.DataFrame]:
    '''
    Transcript loader dataset based on company_id, if nothing found, returns None
    '''
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
    logging.debug('no local transcripts found')
    return None

def run(args: str) -> subprocess.CompletedProcess:
    '''
    Running a subprocess in shell with args parameter (gets split)
    '''
    return subprocess.run(
        args.split(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True
    )

def popen(args) -> subprocess.Popen:
    '''
    Running a subprocess in shell with args parameter (gets split),
    interactive process gets returned, that needs to be terminated.
    '''
    return subprocess.Popen(
        args.split(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

def get_ollama_version():
    '''
    Function to call for Ollama version (for the sake of documentation).
    '''
    try:
        result = run('ollama --version')
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return 'Error while checking Ollama version.'
