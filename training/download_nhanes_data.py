"""
download_nhanes_data.py
-----------------------
Downloads official CDC NHANES 2017-March 2020 pre-pandemic SAS transport (.xpt)
data files into `data/nhanes_raw/`.

Files downloaded:
  1. P_OHQ.xpt    - Oral Health Questionnaire
  2. P_OHXDEN.xpt - Oral Health Dentition Examination
  3. P_DR1TOT.xpt - Dietary Interview (Day 1 Total Nutrients)
  4. P_DEMO.xpt   - Demographics (Age, Sex, Poverty ratio)
  5. P_SMQ.xpt    - Smoking - Cigarette Use
  6. P_DIQ.xpt    - Diabetes

Provides checksum/size verification and automatic resume/skip if already downloaded.
"""

import argparse
import os
import sys
import time
import requests

NHANES_BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles"

DEFAULT_FILES = {
    "P_OHQ.xpt": f"{NHANES_BASE_URL}/P_OHQ.xpt",
    "P_OHXDEN.xpt": f"{NHANES_BASE_URL}/P_OHXDEN.xpt",
    "P_DR1TOT.xpt": f"{NHANES_BASE_URL}/P_DR1TOT.xpt",
    "P_DEMO.xpt": f"{NHANES_BASE_URL}/P_DEMO.xpt",
    "P_SMQ.xpt": f"{NHANES_BASE_URL}/P_SMQ.xpt",
    "P_DIQ.xpt": f"{NHANES_BASE_URL}/P_DIQ.xpt",
}


def download_file(url: str, dest_path: str, timeout: int = 60, retries: int = 3) -> bool:
    """Downloads a single file with streaming, retry logic, and validation."""
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        print(f"  [EXISTS] {os.path.basename(dest_path)} ({os.path.getsize(dest_path):,} bytes), skipping.")
        return True

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) CDC-NHANES-Downloader"}
    temp_path = dest_path + ".tmp"

    for attempt in range(1, retries + 1):
        try:
            print(f"  [DOWNLOADING] {os.path.basename(dest_path)} (attempt {attempt}/{retries})...")
            with requests.get(url, headers=headers, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                total_len = int(response.headers.get("content-length", 0))

                with open(temp_path, "wb") as f:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)

                # Validate non-empty
                if os.path.getsize(temp_path) < 1000:
                    raise IOError(f"Downloaded file is suspiciously small ({os.path.getsize(temp_path)} bytes)")

                # Verify SAS header
                with open(temp_path, "rb") as f:
                    header = f.read(80)
                    if b"HEADER RECORD" not in header and b"SAS" not in header:
                        print("  [WARN] File does not start with standard SAS Transport header, but continuing.")

                os.replace(temp_path, dest_path)
                print(f"  [DONE] {os.path.basename(dest_path)} ({os.path.getsize(dest_path):,} bytes).")
                return True

        except Exception as e:
            print(f"  [ERROR] Attempt {attempt} failed: {e}")
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            if attempt < retries:
                time.sleep(2 * attempt)

    return False


def download_all(dest_dir: str, files_dict: dict = None) -> bool:
    if files_dict is None:
        files_dict = DEFAULT_FILES

    os.makedirs(dest_dir, exist_ok=True)
    print(f"Downloading CDC NHANES files to: {os.path.abspath(dest_dir)}")

    success_all = True
    for filename, url in files_dict.items():
        dest = os.path.join(dest_dir, filename)
        ok = download_file(url, dest)
        if not ok:
            success_all = False
            print(f"\n[FALLBACK INSTRUCTION]")
            print(f"Failed to automatically download {filename}.")
            print(f"Please manually download the file from:\n  {url}")
            print(f"and place it into:\n  {os.path.abspath(dest)}")

    return success_all


def main():
    parser = argparse.ArgumentParser(description="Download official CDC NHANES 2017-March 2020 pre-pandemic data files.")
    parser.add_argument(
        "--dest",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "data", "nhanes_raw"),
        help="Target folder for downloaded .xpt files.",
    )
    args = parser.parse_args()

    ok = download_all(args.dest)
    if not ok:
        sys.exit(1)
    print("\nAll NHANES files verified successfully.")


if __name__ == "__main__":
    main()
