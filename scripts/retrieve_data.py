import urllib.request
import zipfile
from pathlib import Path

URL = "https://dl.dropboxusercontent.com/s/2r19631jo062ci7/data.zip"
PATH = Path(".")

if __name__ == "__main__":
    print("downloading `data.zip`...")
    zip_path, _ = urllib.request.urlretrieve(URL)

    print("extracting `data.zip`...")
    with zipfile.ZipFile(zip_path, "r") as f:
        f.extractall(PATH)

    print("done")
