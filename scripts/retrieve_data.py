import urllib.request
import zipfile
from pathlib import Path

URL = "https://www.dropbox.com/sh/d2ivdhq30xhcky1/AAD_ajkFDfV6L7AlAI3amtTya?dl=1"
PATH = Path("./data")

if __name__ == "__main__":
    print("creating data directory...")
    PATH.mkdir(exist_ok=True)

    print("downloading the data...")
    zip_path, _ = urllib.request.urlretrieve(URL)

    print("extracting the data...")
    with zipfile.ZipFile(zip_path, "r") as f:
        f.extractall(PATH)

    print("done")
