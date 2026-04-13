"""
Collect State of the Union addresses (1950-2026).
Speeches 1950-2020 from the stdlib-js/datasets-sotu public GitHub repository.
Speeches 2021-2026 from the American Presidency Project (presidency.ucsb.edu).
"""

import json
import os
import csv
import urllib.request
import time

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
BASE_URL = "https://raw.githubusercontent.com/stdlib-js/datasets-sotu/main/data/"

# Files available in the repository for 1950-2020
SOTU_FILES = [
    ("1950_harry_s_truman_d.txt", 1950, "Harry S. Truman", "Democrat"),
    ("1951_harry_s_truman_d.txt", 1951, "Harry S. Truman", "Democrat"),
    ("1952_harry_s_truman_d.txt", 1952, "Harry S. Truman", "Democrat"),
    ("1953_harry_s_truman_d.txt", 1953, "Harry S. Truman", "Democrat"),
    ("1953_dwight_d_eisenhower_r.txt", 1953, "Dwight D. Eisenhower", "Republican"),
    ("1954_dwight_d_eisenhower_r.txt", 1954, "Dwight D. Eisenhower", "Republican"),
    ("1955_dwight_d_eisenhower_r.txt", 1955, "Dwight D. Eisenhower", "Republican"),
    ("1956_dwight_d_eisenhower_r.txt", 1956, "Dwight D. Eisenhower", "Republican"),
    ("1957_dwight_d_eisenhower_r.txt", 1957, "Dwight D. Eisenhower", "Republican"),
    ("1958_dwight_d_eisenhower_r.txt", 1958, "Dwight D. Eisenhower", "Republican"),
    ("1959_dwight_d_eisenhower_r.txt", 1959, "Dwight D. Eisenhower", "Republican"),
    ("1960_dwight_d_eisenhower_r.txt", 1960, "Dwight D. Eisenhower", "Republican"),
    ("1961_dwight_d_eisenhower_r.txt", 1961, "Dwight D. Eisenhower", "Republican"),
    ("1961_john_f_kennedy_d.txt", 1961, "John F. Kennedy", "Democrat"),
    ("1962_john_f_kennedy_d.txt", 1962, "John F. Kennedy", "Democrat"),
    ("1963_john_f_kennedy_d.txt", 1963, "John F. Kennedy", "Democrat"),
    ("1964_lyndon_b_johnson_d.txt", 1964, "Lyndon B. Johnson", "Democrat"),
    ("1965_lyndon_b_johnson_d.txt", 1965, "Lyndon B. Johnson", "Democrat"),
    ("1966_lyndon_b_johnson_d.txt", 1966, "Lyndon B. Johnson", "Democrat"),
    ("1967_lyndon_b_johnson_d.txt", 1967, "Lyndon B. Johnson", "Democrat"),
    ("1968_lyndon_b_johnson_d.txt", 1968, "Lyndon B. Johnson", "Democrat"),
    ("1969_lyndon_b_johnson_d.txt", 1969, "Lyndon B. Johnson", "Democrat"),
    ("1970_richard_nixon_r.txt", 1970, "Richard Nixon", "Republican"),
    ("1971_richard_nixon_r.txt", 1971, "Richard Nixon", "Republican"),
    ("1972_richard_nixon_r.txt", 1972, "Richard Nixon", "Republican"),
    ("1973_richard_nixon_r.txt", 1973, "Richard Nixon", "Republican"),
    ("1974_richard_nixon_r.txt", 1974, "Richard Nixon", "Republican"),
    ("1975_gerald_r_ford_r.txt", 1975, "Gerald R. Ford", "Republican"),
    ("1976_gerald_r_ford_r.txt", 1976, "Gerald R. Ford", "Republican"),
    ("1977_gerald_r_ford_r.txt", 1977, "Gerald R. Ford", "Republican"),
    ("1978_jimmy_carter_d.txt", 1978, "Jimmy Carter", "Democrat"),
    ("1979_jimmy_carter_d.txt", 1979, "Jimmy Carter", "Democrat"),
    ("1980_jimmy_carter_d.txt", 1980, "Jimmy Carter", "Democrat"),
    ("1981_jimmy_carter_d.txt", 1981, "Jimmy Carter", "Democrat"),
    ("1982_ronald_reagan_r.txt", 1982, "Ronald Reagan", "Republican"),
    ("1983_ronald_reagan_r.txt", 1983, "Ronald Reagan", "Republican"),
    ("1984_ronald_reagan_r.txt", 1984, "Ronald Reagan", "Republican"),
    ("1985_ronald_reagan_r.txt", 1985, "Ronald Reagan", "Republican"),
    ("1986_ronald_reagan_r.txt", 1986, "Ronald Reagan", "Republican"),
    ("1987_ronald_reagan_r.txt", 1987, "Ronald Reagan", "Republican"),
    ("1988_ronald_reagan_r.txt", 1988, "Ronald Reagan", "Republican"),
    ("1989_george_bush_r.txt", 1989, "George H.W. Bush", "Republican"),
    ("1990_george_bush_r.txt", 1990, "George H.W. Bush", "Republican"),
    ("1991_george_bush_r.txt", 1991, "George H.W. Bush", "Republican"),
    ("1992_george_bush_r.txt", 1992, "George H.W. Bush", "Republican"),
    ("1993_william_j_clinton_d.txt", 1993, "Bill Clinton", "Democrat"),
    ("1994_william_j_clinton_d.txt", 1994, "Bill Clinton", "Democrat"),
    ("1995_william_j_clinton_d.txt", 1995, "Bill Clinton", "Democrat"),
    ("1996_william_j_clinton_d.txt", 1996, "Bill Clinton", "Democrat"),
    ("1997_william_j_clinton_d.txt", 1997, "Bill Clinton", "Democrat"),
    ("1998_william_j_clinton_d.txt", 1998, "Bill Clinton", "Democrat"),
    ("1999_william_j_clinton_d.txt", 1999, "Bill Clinton", "Democrat"),
    ("2000_william_j_clinton_d.txt", 2000, "Bill Clinton", "Democrat"),
    ("2001_george_w_bush_r.txt", 2001, "George W. Bush", "Republican"),
    ("2002_george_w_bush_r.txt", 2002, "George W. Bush", "Republican"),
    ("2003_george_w_bush_r.txt", 2003, "George W. Bush", "Republican"),
    ("2004_george_w_bush_r.txt", 2004, "George W. Bush", "Republican"),
    ("2005_george_w_bush_r.txt", 2005, "George W. Bush", "Republican"),
    ("2006_george_w_bush_r.txt", 2006, "George W. Bush", "Republican"),
    ("2007_george_w_bush_r.txt", 2007, "George W. Bush", "Republican"),
    ("2008_george_w_bush_r.txt", 2008, "George W. Bush", "Republican"),
    ("2009_barack_obama_d.txt", 2009, "Barack Obama", "Democrat"),
    ("2010_barack_obama_d.txt", 2010, "Barack Obama", "Democrat"),
    ("2011_barack_obama_d.txt", 2011, "Barack Obama", "Democrat"),
    ("2012_barack_obama_d.txt", 2012, "Barack Obama", "Democrat"),
    ("2013_barack_obama_d.txt", 2013, "Barack Obama", "Democrat"),
    ("2014_barack_obama_d.txt", 2014, "Barack Obama", "Democrat"),
    ("2015_barack_obama_d.txt", 2015, "Barack Obama", "Democrat"),
    ("2016_barack_obama_d.txt", 2016, "Barack Obama", "Democrat"),
    ("2017_donald_j_trump_r.txt", 2017, "Donald J. Trump", "Republican"),
    ("2018_donald_j_trump_r.txt", 2018, "Donald J. Trump", "Republican"),
    ("2019_donald_j_trump_r.txt", 2019, "Donald J. Trump", "Republican"),
    ("2020_donald_j_trump_r.txt", 2020, "Donald J. Trump", "Republican"),
    ("2021_joseph_r_biden_d.txt", 2021, "Joseph R. Biden", "Democrat"),
    ("2022_joseph_r_biden_d.txt", 2022, "Joseph R. Biden", "Democrat"),
    ("2023_joseph_r_biden_d.txt", 2023, "Joseph R. Biden", "Democrat"),
    ("2024_joseph_r_biden_d.txt", 2024, "Joseph R. Biden", "Democrat"),
    ("2025_donald_j_trump_r.txt", 2025, "Donald J. Trump", "Republican"),
    ("2026_donald_j_trump_r.txt", 2026, "Donald J. Trump", "Republican"),
]


def download_speeches():
    """Download all SOTU speeches from GitHub."""
    os.makedirs(DATA_DIR, exist_ok=True)
    texts_dir = os.path.join(DATA_DIR, 'texts')
    os.makedirs(texts_dir, exist_ok=True)

    speeches = []
    for filename, year, president, party in SOTU_FILES:
        url = BASE_URL + filename
        out_path = os.path.join(texts_dir, filename)

        if os.path.exists(out_path):
            with open(out_path, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()
            print(f"  [cached] {filename}")
        else:
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    text = resp.read().decode('utf-8', errors='replace')
                with open(out_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                print(f"  [downloaded] {filename} ({len(text)} chars)")
                time.sleep(0.2)  # be polite to GitHub
            except Exception as e:
                print(f"  [FAILED] {filename}: {e}")
                continue

        speeches.append({
            'year': year,
            'president': president,
            'party': party,
            'filename': filename,
            'text': text,
            'word_count': len(text.split())
        })

    return speeches


def save_corpus(speeches):
    """Save corpus as JSON and metadata CSV."""
    # Save full corpus as JSON
    json_path = os.path.join(DATA_DIR, 'sotu_corpus.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(speeches, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(speeches)} speeches to {json_path}")

    # Save metadata CSV
    csv_path = os.path.join(DATA_DIR, 'sotu_metadata.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['year', 'president', 'party', 'filename', 'word_count'])
        writer.writeheader()
        for s in speeches:
            writer.writerow({k: s[k] for k in ['year', 'president', 'party', 'filename', 'word_count']})
    print(f"Saved metadata to {csv_path}")


if __name__ == '__main__':
    print("Downloading SOTU speeches (1950-2020)...")
    speeches = download_speeches()
    save_corpus(speeches)
    print(f"\nDone: {len(speeches)} speeches collected")
