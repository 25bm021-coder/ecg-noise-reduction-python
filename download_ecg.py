
from pathlib import Path
import urllib.request
import time

data_folder = Path("data")
data_folder.mkdir(exist_ok=True)

base_url = "https://physionet.org/files/mitdb/1.0.0/"

files = ["100.hea", "100.dat", "100.atr"]

for filename in files:
    destination = data_folder / filename
    url = base_url + filename

    if destination.exists() and destination.stat().st_size > 0:
        print(f"Already downloaded: {filename}", flush=True)
        continue

    print(f"Downloading {filename}...", flush=True)

    for attempt in range(3):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "ECG-AI-Student-Project"}
            )

            with urllib.request.urlopen(request, timeout=60) as response:
                total = int(response.headers.get("Content-Length", 0))
                downloaded = 0

                with open(destination, "wb") as output:
                    while True:
                        chunk = response.read(64 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
                        downloaded += len(chunk)

                        if total:
                            percent = downloaded * 100 // total
                            print(
                                f"\r{filename}: {percent}%",
                                end="",
                                flush=True
                            )

            print(f"\nCompleted: {filename}", flush=True)
            break

        except Exception as error:
            print(f"\nAttempt {attempt + 1} failed: {error}", flush=True)
            destination.unlink(missing_ok=True)

            if attempt == 2:
                raise

            time.sleep(2)

print("\nECG dataset download completed!")
print("Files saved in:", data_folder.resolve())
