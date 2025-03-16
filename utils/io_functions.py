'''
Module for using shell and file system (mostly csv files).
'''

import logging
# import os
from typing import Optional
from pathlib import Path
import subprocess
import zipfile
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

def zip_directory(directory_path, zip_filename):
    '''
    Zipping a directory with Linux-style paths, but the zip file should be compatible with Windows 
    (i.e., line endings in text files should be Windows-style \r\n).
    '''

    directory_path = Path(directory_path)

    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in directory_path.rglob('*'):
            if file_path.is_file():
                relpath = file_path.relative_to(directory_path).as_posix()
                with open(file_path, 'rb') as f:
                    file_data = f.read()
                if b'\n' in file_data:
                    file_data = file_data.replace(b'\n', b'\r\n')
                zipf.writestr(relpath, file_data)

# if __name__ == "__main__":
#     # Define the directory and zip filename
#     directory_to_zip = "../data"
#     zip_file_name = "../data/data.zip"
#     # Call the function to zip the directory
#     zip_directory(directory_to_zip, zip_file_name)
#     print(f"Directory '{directory_to_zip}' has been zipped into '{zip_file_name}'")
