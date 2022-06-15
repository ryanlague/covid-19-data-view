
from pathlib import Path
import requests
from tqdm import tqdm


def downloadFromUrl(url, outPath):
    # The name of the file
    filename = Path(url).name
    # HTTP Request
    r = requests.get(url, stream=True)

    # Stream the file
    chunk_size = 1024
    with open(outPath, 'wb') as f:
        total_length = int(r.headers.get('content-length', 0))
        total_size = (total_length / chunk_size) + 1
        for chunk in tqdm(r.iter_content(chunk_size=chunk_size), total=total_size, desc=f'Downloading {filename}'):
            if chunk:
                f.write(chunk)
                f.flush()


if __name__ == '__main__':
    # Can be downloaded manually from https://github.com/owid/covid-19-data/tree/master/public/data/owid-covid-data.csv
    URL = 'https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv'

    # Save the file here
    out_dir = Path('data')
    out_path = out_dir.joinpath(Path(URL).name)

    # Download
    downloadFromUrl(URL, out_path)
