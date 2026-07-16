"""
Incremental CSV metric logging.
One row per call to `log`, appended as it goes — the caller decides the cadence.
Resuming an existing run appends rather than truncating.
"""

import csv
import os


class CSVLogger:
    def __init__(self, path: str, fields: list):
        self.path = path
        self.fields = list(fields)
        is_new = not os.path.exists(path) or os.path.getsize(path) == 0
        self._file = open(path, "a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.fields)
        if is_new:
            self._writer.writeheader()
            self._file.flush()

    def log(self, **row):
        unknown = set(row) - set(self.fields)
        if unknown:
            raise ValueError(f"unknown log fields {sorted(unknown)}; expected {self.fields}.")
        self._writer.writerow({k: row.get(k, "") for k in self.fields})
        self._file.flush()

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
