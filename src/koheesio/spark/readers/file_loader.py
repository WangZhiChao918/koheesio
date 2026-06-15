"""
Generic file Readers for different file formats.

Supported file formats:
- CSV
- Parquet
- Avro
- JSON
- ORC
- Text

Examples:
```python
from koheesio.spark.readers import (
    CsvReader,
    ParquetReader,
    AvroReader,
    JsonReader,
    OrcReader,
)

csv_reader = CsvReader(path="path/to/file.csv", header=True)
parquet_reader = ParquetReader(path="path/to/file.parquet")
avro_reader = AvroReader(path="path/to/file.avro")
json_reader = JsonReader(path="path/to/file.json")
orc_reader = OrcReader(path="path/to/file.orc")
```

For more information about the available options, see Spark's
[official documentation](https://spark.apache.org/docs/latest/sql-data-sources.html).
"""

import re
from typing import Optional, Union
from enum import Enum
from glob import glob
from pathlib import Path

from pyspark.sql.types import StructType

from koheesio.models import ExtraParamsMixin, Field, field_validator
from koheesio.spark.readers import Reader


class FileFormat(str, Enum):
    """Supported file formats.

    This enum represents the supported file formats that can be used with the FileLoader class.
    The available file formats are:
    - csv: Comma-separated values format
    - parquet: Apache Parquet format
    - avro: Apache Avro format
    - json: JavaScript Object Notation format
    - orc: Apache ORC format
    - text: Plain text format
    """

    csv = "csv"
    parquet = "parquet"
    avro = "avro"
    json = "json"
    orc = "orc"
    # excel = "excel"  # TODO: Add support for Excel
    # xml = "xml"  # TODO: Add support for XML
    # yaml = "yaml"  # TODO: Add support for YAML
    text = "text"


# Matches a URI scheme such as ``s3://``, ``s3a://``, ``gs://``, ``hdfs://``, ``abfss://`` ...
_URI_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")

# Glob metacharacters understood by ``glob.glob``.
_GLOB_MAGIC_RE = re.compile(r"[*?\[]")

# Binary / columnar formats. A mismatch involving one of these is almost always a genuine
# error (e.g. reading a CSV file with the Parquet reader produces a cryptic Spark error),
# whereas text-based formats (csv/json/text) are frequently interchangeable.
_BINARY_FORMATS = frozenset({FileFormat.parquet, FileFormat.avro, FileFormat.orc})

# Maps a file extension to the format it most likely represents.
_EXTENSION_TO_FORMAT = {
    ".csv": FileFormat.csv,
    ".json": FileFormat.json,
    ".txt": FileFormat.text,
    ".parquet": FileFormat.parquet,
    ".avro": FileFormat.avro,
    ".orc": FileFormat.orc,
}


def _is_local_path(path: str) -> bool:
    """Return whether ``path`` points at the local filesystem and is safe to inspect.

    Remote/URI paths (``s3://``, ``hdfs://``, ``dbfs:/`` ...) and comma-separated multi-paths
    are intentionally treated as *not* local so that valid Spark reads are never altered.
    """
    if "," in path:
        # Spark accepts a comma-separated list of paths; leave those to Spark.
        return False
    if path.startswith(("dbfs:", "file:")):
        return False
    if _URI_SCHEME_RE.match(path):
        return False
    return True


def _has_glob_magic(path: str) -> bool:
    """Return whether ``path`` contains glob metacharacters (``*``, ``?`` or ``[``)."""
    return bool(_GLOB_MAGIC_RE.search(path))


# pylint: disable=line-too-long
class FileLoader(Reader, ExtraParamsMixin):
    """Generic file reader.

    Available file formats:
    - CSV
    - Parquet
    - Avro
    - JSON
    - ORC
    - Text (default)

    Extra parameters can be passed to the reader using the `extra_params` attribute or as keyword arguments.

    Example:
    ```python
    reader = FileLoader(
        path="path/to/textfile.txt",
        format="text",
        header=True,
        lineSep="\n",
    )
    ```

    For more information about the available options, see Spark's
    [official pyspark documentation](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrameReader.text.html)
    and [read about text data source](https://spark.apache.org/docs/latest/sql-data-sources-text.html).

    Also see the [data sources generic options](https://spark.apache.org/docs/3.5.0/sql-data-sources-generic-options.html).
    """

    format: FileFormat = Field(default=FileFormat.text, description="File format to read")
    path: Union[Path, str] = Field(default=..., description="Path to the file to read")
    schema_: Optional[Union[StructType, str]] = Field(
        default=None, description="Schema to use when reading the file", validate_default=False, alias="schema"
    )
    streaming: Optional[bool] = Field(default=False, description="Whether to read the files as a Stream or not")

    @field_validator("path")
    def ensure_path_is_str(cls, path: Union[Path, str]) -> Union[Path, str]:
        """Ensure that the path is a string as required by Spark."""
        if isinstance(path, Path):
            return str(path.absolute().as_posix())
        return path

    def _check_extension(self, file_path: str) -> None:
        """Raise a clear ``ValueError`` when a single file's extension conflicts with ``format``.

        The check is deliberately lenient: the ``text`` reader accepts any file, and text-based
        formats (csv/json/txt) are treated as interchangeable. A mismatch is only reported when a
        binary/columnar format (parquet/avro/orc) is involved on either side, since those are the
        cases where Spark would otherwise fail with a hard-to-interpret error.
        """
        if self.format == FileFormat.text:
            return

        suffix = Path(file_path).suffix.lower()
        ext_format = _EXTENSION_TO_FORMAT.get(suffix)
        if ext_format is None or ext_format == self.format:
            return

        if self.format in _BINARY_FORMATS or ext_format in _BINARY_FORMATS:
            # `use_enum_values=True` means `self.format` may be a plain str; normalise for display.
            fmt = getattr(self.format, "value", self.format)
            raise ValueError(
                f"File extension '{suffix}' (which maps to format '{ext_format.value}') does not "
                f"match the reader format '{fmt}' for path '{file_path}'. "
                f"Verify the file format, or set the reader's `format` explicitly."
            )

    def _validate_and_discover_path(self) -> None:
        """Surface clearer errors for broken local paths before handing off to Spark.

        Only local filesystem paths are inspected. Remote/URI paths (``s3://``, ``hdfs://`` ...)
        and comma-separated multi-paths are left untouched so that valid reads are unaffected.

        Raises:
            FileNotFoundError: when a concrete local path does not exist, or a glob pattern matches
                no files. The message includes the path/pattern, the ``format`` and (for globs) the
                number of matches.
            ValueError: when a single local file's extension conflicts with the reader ``format``.
        """
        path = str(self.path)
        if not _is_local_path(path):
            return

        # `use_enum_values=True` means `self.format` may be a plain str; normalise for messages.
        fmt = getattr(self.format, "value", self.format)

        if _has_glob_magic(path):
            matches = glob(path)
            if not matches:
                raise FileNotFoundError(
                    f"No files matched the glob pattern '{path}' (format='{fmt}', "
                    f"matches=0). Check that the pattern is correct and that the files exist."
                )
            return

        resolved = Path(path)
        if not resolved.exists():
            raise FileNotFoundError(
                f"Path does not exist: '{path}' (format='{fmt}'). "
                f"Check that the path is correct and accessible."
            )

        # Directories are valid containers for partitioned data (parquet/orc/avro/delta); only
        # validate the extension of concrete single files.
        if resolved.is_file():
            self._check_extension(path)

    def execute(self) -> Reader.Output:
        """Reads the file, in batch or as a stream, using the specified format and schema, while applying any extra parameters."""
        self._validate_and_discover_path()
        reader = self.spark.readStream if self.streaming else self.spark.read
        reader = reader.format(self.format)

        if self.schema_:
            reader.schema(self.schema_)

        if self.extra_params:
            reader = reader.options(**self.extra_params)

        self.output.df = reader.load(self.path)  # type: ignore


class CsvReader(FileLoader):
    """Reads a CSV file.

    This class is a convenience class that sets the `format` field to `FileFormat.csv`.

    Extra parameters can be passed to the reader using the `extra_params` attribute or as keyword arguments.

    Example:
    ```python
    reader = CsvReader(path="path/to/file.csv", header=True)
    ```

    For more information about the available options,
    see the
    [official pyspark documentation](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrameReader.csv.html)
    and [read about CSV data source](https://spark.apache.org/docs/latest/sql-data-sources-csv.html).

    Also see the
    [data sources generic options](https://spark.apache.org/docs/latest/sql-data-sources-generic-options.html).
    """

    format: FileFormat = FileFormat.csv


class ParquetReader(FileLoader):
    """Reads a Parquet file.

    This class is a convenience class that sets the `format` field to `FileFormat.parquet`.

    Extra parameters can be passed to the reader using the `extra_params` attribute or as keyword arguments.

    Example:
    ```python
    reader = ParquetReader(path="path/to/file.parquet", mergeSchema=True)
    ```

    For more information about the available options,
    see the
    [official pyspark documentation](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrameReader.parquet.html)
    and [read about Parquet data source](https://spark.apache.org/docs/latest/sql-data-sources-parquet.html).

    Also see the
    [data sources generic options](https://spark.apache.org/docs/latest/sql-data-sources-generic-options.html).
    """

    format: FileFormat = FileFormat.parquet


class AvroReader(FileLoader):
    """Reads an Avro file.

    This class is a convenience class that sets the `format` field to `FileFormat.avro`.

    Extra parameters can be passed to the reader using the `extra_params` attribute or as keyword arguments.

    Example:
    ```python
    reader = AvroReader(path="path/to/file.avro", mergeSchema=True)
    ```

    Make sure to have the `spark-avro` package installed in your environment.

    For more information about the available options,
    see the [official documentation](https://spark.apache.org/docs/latest/sql-data-sources-avro.html#content).
    """

    format: FileFormat = FileFormat.avro


class JsonReader(FileLoader):
    """Reads a JSON file.

    This class is a convenience class that sets the `format` field to `FileFormat.json`.

    Extra parameters can be passed to the reader using the `extra_params` attribute or as keyword arguments.

    Example:
    ```python
    reader = JsonReader(path="path/to/file.json", allowComments=True)
    ```

    For more information about the available options,
    see the
    [official pyspark documentation](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrameReader.json.html#pyspark.sql.DataFrameReader.json)
    and [read about JSON data source](https://spark.apache.org/docs/latest/sql-data-sources-json.html).

    Also see the
    [data sources generic options](https://spark.apache.org/docs/latest/sql-data-sources-generic-options.html).
    """

    format: FileFormat = FileFormat.json


class OrcReader(FileLoader):
    """Reads an ORC file.

    This class is a convenience class that sets the `format` field to `FileFormat.orc`.

    Extra parameters can be passed to the reader using the `extra_params` attribute or as keyword arguments.

    Example:
    ```python
    reader = OrcReader(path="path/to/file.orc", mergeSchema=True)
    ```

    For more information about the available options,
    see the
    [official documentation](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrameReader.orc.html)
    and [read about ORC data source](https://spark.apache.org/docs/latest/sql-data-sources-orc.html).

    Also see the
    [data sources generic options](https://spark.apache.org/docs/latest/sql-data-sources-generic-options.html).
    """

    format: FileFormat = FileFormat.orc
