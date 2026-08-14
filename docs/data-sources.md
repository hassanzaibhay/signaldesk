# Data sources

Provenance, licence terms, and the practical properties of each source. The
awkward details are recorded here deliberately: every one of them cost real time
to find, and none of them is in the upstream documentation.

## FDA adverse event quarterly extracts

**Landing page:** <https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html>
**Archives:** `https://fis.fda.gov/content/Exports/faers_ascii_<year><quarter>.zip`
**Range ingested:** 2012Q4 to the latest published quarter (2026Q2 as of 2026-08-13).
**Licence:** work of the United States Government, public domain. No attribution
is required; it is given anyway.
**Refresh:** quarterly, roughly two months after the quarter closes.

Legacy AERS extracts (2004 to 2012Q3) use an `aers_ascii_` prefix and a report-based
schema with no case version, and are out of scope.

### The archive URLs cannot be constructed

Observed 2026-08-13: **18 of the 55 quarters capitalise the quarter letter** and
37 do not.

```
faers_ascii_2013q1.zip     lower case
faers_ascii_2019Q1.zip     upper case
faers_ascii_2026q2.zip     lower case
```

Upper case: 2019Q1-Q4, 2020Q1-Q4, 2021Q1-Q4, 2022Q3, 2022Q4, 2023Q3, 2023Q4,
2024Q4, 2025Q4. There is no pattern to it and no reason to expect the next one
to follow either convention.

`ingest/faers/discover.py` therefore scrapes the landing page and keeps the href
it actually finds. Constructing a URL from a quarter is a fallback only, and logs
a warning when it is used. Do not "simplify" this into an f-string.

### The server needs specific handling

Three properties, all measured rather than documented:

1. **`Accept-Encoding: identity` is mandatory.** With a normal content
   negotiation header the server accepts the connection and then never responds,
   so the request dies at the read timeout. Asking for no encoding returns the
   bytes immediately. This is why the download path sets the header explicitly.
   A client that works fine against every other host will appear to hang here.
2. **Connections drop mid-body.** A response will stop early with
   `peer closed connection without sending complete message body`. Segments are
   therefore re-requested from where they stopped rather than restarted.
3. **The server is slow to answer, not only slow to transfer.** A one-byte ranged
   GET has taken over 30 seconds to produce headers, so this source uses longer
   timeouts than the shared client's defaults.

`HEAD` returns `Content-Length: 0`, so the file size comes from the
`Content-Range` of a one-byte ranged request. Range requests are supported and
throughput scales with connections; four are used.

### Archive layout varies between quarters

* The data directory is `ascii/` in some quarters and `ASCII/` in others.
* Line endings are CRLF in some files and LF in others, **within the same
  archive**: in 2026Q2, `DRUG26Q2.txt` is CRLF and `DEMO26Q2.txt` is LF.
* Encoding is per file, not per archive: `DRUG13Q1.txt` is UTF-8 while its
  siblings are ASCII. `drug12q4.txt` additionally carries a byte order mark.
* Some quarters pad header field names with spaces.
* Fields are `$`-separated and **not quoted**. Drug names contain bare quote
  characters, so quote processing must be disabled or rows silently merge.

### Deleted-cases lists

Later quarters ship `Deleted/DELETE<quarter>.txt`, a bare list of withdrawn case
identifiers with no header whose **first line is a single space**. Earlier
quarters ship no such file: 2012Q4, all four 2014 quarters and 2013Q1 have none,
while 2026Q2 does.

An absent file is treated as an empty set, but absent and empty are not the same
claim, so the quality report lists which quarters shipped no file.

### Three schema eras

Derived by reading headers, not from documentation.

| Era | Quarters | Distinguishing columns |
|---|---|---|
| `2012q4_only` | 2012Q4 | `lot_nbr` in DRUG, `outc_code` in OUTC |
| `pre_2014q3` | 2013Q1 to 2014Q2 | `gndr_cod` in DEMO, DRUG 19 cols, REAC 3 cols |
| `2014q3_onward` | 2014Q3 onward | `sex` in DEMO, `prod_ai` in DRUG, `drug_rec_act` in REAC |

The boundary was bracketed empirically: 2014Q2 has 22/19/3 columns for
DEMO/DRUG/REAC and 2014Q3 has 25/20/4.

`prod_ai`, the active ingredient the source supplies, exists only from 2014Q3.
It is null for every earlier row by construction rather than by data quality,
which bounds how far drug normalization can lean on it.

### Age unit codes

Eight, not the six the documentation implies: `DEC`, `YR`, `MON`, `WK`, `DY`,
`HR`, `MIN`, `SEC`. `SEC` appears in 2012Q4 and was found only because the
parser refuses to guess at an unrecognised code.

### Report sources are ingested but not modelled

`RPSR` ships in every archive and is written to Parquet as the `report_source`
dataset. It has no Postgres model and no consumer in the roadmap; it is available
through `signaldesk.analytics.faers` if one appears later. It was not missed.

### Where the data lives

Cases, duplicate judgements and the ingest manifest are in Postgres. The five
high-cardinality child datasets - drug, reaction, outcome, therapy, indication -
plus report sources are held in Parquet and read through DuckDB. See
`docs/adr/0004` for the measurements behind that split.
