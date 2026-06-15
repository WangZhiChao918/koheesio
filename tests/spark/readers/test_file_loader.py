import pytest

import pyspark.sql.types as T

from koheesio.spark import AnalysisException
from koheesio.spark.readers.file_loader import (
    AvroReader,
    CsvReader,
    FileFormat,
    FileLoader,
    JsonReader,
    OrcReader,
    ParquetReader,
    _is_local_path,
)

pytestmark = pytest.mark.spark


@pytest.fixture()
def json_file(data_path):
    return f"{data_path}/readers/json_file/dummy_simple.json"


@pytest.fixture()
def csv_comma_file(data_path):
    return f"{data_path}/readers/csv_file/dummy.csv"


@pytest.fixture()
def csv_semicolon_file(data_path):
    return f"{data_path}/readers/csv_file/dummy_semicolon.csv"


@pytest.fixture()
def parquet_file(data_path):
    return f"{data_path}/readers/delta_file"


@pytest.fixture()
def avro_file(data_path):
    return f"{data_path}/readers/avro_file"


@pytest.fixture()
def orc_file(data_path):
    return f"{data_path}/readers/orc_file"


def test_file_loader(csv_comma_file):
    # test schema
    expected_data = [
        {"value": "string,int,float"},
        {"value": "string1,1,1.0"},
        {"value": "string2,2,2.0"},
        {"value": "string3,3,3.0"},
    ]
    schema = "value STRING"
    reader = FileLoader(path=csv_comma_file, header=True, schema=schema, lineSep="\n")
    assert reader.schema_ == schema
    df = reader.read()
    actual_data = [row.asDict() for row in df.collect()]
    print(f"{actual_data = }")
    assert actual_data == expected_data


def test_invalid_file_format(csv_comma_file):
    with pytest.raises(ValueError):
        FileLoader(format="invalid_format", path=csv_comma_file, header=True)


def test_csv_reader(csv_comma_file, csv_semicolon_file):
    expected_data = [
        {"string": "string1", "int": 1, "float": 1.0},
        {"string": "string2", "int": 2, "float": 2.0},
        {"string": "string3", "int": 3, "float": 3.0},
    ]

    schema = "string STRING, int INT, float FLOAT"

    # comma separated file
    reader = CsvReader(path=csv_comma_file, header=True, schema=schema)
    assert reader.path == csv_comma_file
    assert reader.header is True
    assert reader.format == FileFormat.csv.value
    df = reader.read()
    actual_data = [row.asDict() for row in df.collect()]
    assert actual_data == expected_data

    # semicolon separated file
    reader = CsvReader(path=csv_semicolon_file, header=True, sep=";", schema=schema)
    assert reader.path == csv_semicolon_file
    df = reader.read()
    actual_data = [row.asDict() for row in df.collect()]
    assert actual_data == expected_data


def test_json_reader(json_file):
    expected_data = [
        {"string": "string1", "int": 1, "float": 1.0},
        {"string": "string2", "int": 2, "float": 2.0},
        {"string": "string3", "int": 3, "float": 3.0},
    ]

    reader = JsonReader(path=json_file)
    assert reader.path == json_file
    df = reader.read()
    actual_data = [row.asDict() for row in df.collect()]
    assert actual_data == expected_data


def test_json_stream_reader(json_file):
    schema = "string STRING, int INT, float FLOAT"
    reader = JsonReader(path=json_file, schema=schema, streaming=True)
    assert reader.path == json_file
    df = reader.read()
    assert df.isStreaming
    assert df.schema == T._parse_datatype_string(schema)


def test_parquet_reader(parquet_file):
    expected_data = [
        {"id": 0},
        {"id": 1},
        {"id": 2},
    ]

    reader = ParquetReader(path=parquet_file)
    assert reader.path == parquet_file
    df = reader.read()
    actual_data = [row.asDict() for row in df.collect()[0:3]]
    assert actual_data == expected_data


def test_avro_reader(avro_file):
    expected_data = [
        {"string": "string1", "int": 1, "float": 1.0},
        {"string": "string2", "int": 2, "float": 2.0},
        {"string": "string3", "int": 3, "float": 3.0},
    ]

    reader = AvroReader(path=avro_file)

    try:
        assert reader.path == avro_file
        df = reader.read()
        actual_data = [row.asDict() for row in df.collect()]
        assert actual_data == expected_data
    except AnalysisException as e:
        # Avro is not always supported in all environments
        reader.log.error(e)
        assert True


def test_orc_reader(orc_file):
    expected_data = [
        {"string": "string1", "int": 1, "float": 1.0},
        {"string": "string2", "int": 2, "float": 2.0},
        {"string": "string3", "int": 3, "float": 3.0},
    ]

    reader = OrcReader(path=orc_file)
    assert reader.path == orc_file
    df = reader.read()
    actual_data = [row.asDict() for row in df.collect()]
    assert actual_data == expected_data


def test_path_does_not_exist_raises_clear_error(data_path):
    """A non-existent local path should raise a clear FileNotFoundError with path + format."""
    missing = f"{data_path}/readers/csv_file/does_not_exist.csv"
    reader = CsvReader(path=missing, header=True)

    with pytest.raises(FileNotFoundError) as excinfo:
        reader.read()

    msg = str(excinfo.value)
    assert "does_not_exist.csv" in msg
    assert "format='csv'" in msg


def test_glob_pattern_no_match_raises_clear_error(data_path):
    """A glob pattern that matches nothing should report the pattern, format, and matches=0."""
    pattern = f"{data_path}/readers/csv_file/*.tsv"
    reader = CsvReader(path=pattern, header=True)

    with pytest.raises(FileNotFoundError) as excinfo:
        reader.read()

    msg = str(excinfo.value)
    assert pattern in msg
    assert "matches=0" in msg
    assert "format='csv'" in msg


def test_extension_mismatch_raises_clear_error(csv_comma_file):
    """Reading a .csv file with the Parquet reader should fail early with a clear ValueError."""
    reader = ParquetReader(path=csv_comma_file)

    with pytest.raises(ValueError) as excinfo:
        reader.read()

    msg = str(excinfo.value)
    assert ".csv" in msg
    assert "parquet" in msg


def test_glob_extension_mismatch_raises_clear_error(data_path):
    """A glob whose matched files conflict with a binary reader format should fail early.

    The error must name the conflicting extension, the reader format, and the originating
    glob pattern so the user can tell *which* pattern produced the mismatch.
    """
    pattern = f"{data_path}/readers/json_file/*.json"
    reader = ParquetReader(path=pattern)

    with pytest.raises(ValueError) as excinfo:
        reader.read()

    msg = str(excinfo.value)
    assert ".json" in msg
    assert "parquet" in msg
    assert pattern in msg


def test_glob_pattern_with_match_reads_successfully(data_path):
    """A glob that resolves to files must pass discovery and read normally (unaffected)."""
    expected_data = [
        {"string": "string1", "int": 1, "float": 1.0},
        {"string": "string2", "int": 2, "float": 2.0},
        {"string": "string3", "int": 3, "float": 3.0},
    ]
    # Matches only dummy_simple.json (not dummy.json) in the json_file directory.
    pattern = f"{data_path}/readers/json_file/dummy_s*.json"
    reader = JsonReader(path=pattern)

    df = reader.read()
    actual_data = [row.asDict() for row in df.collect()]
    assert actual_data == expected_data


def test_recursive_glob_discovers_nested_files(data_path):
    """A recursive ``**`` pattern must discover files nested several directories deep.

    Regression guard for lake-style ingestion: local discovery globs with ``recursive=True`` so a
    pattern spanning multiple partition directories is not falsely reported as ``matches=0``. The
    only ``*0.json`` file under ``readers/`` lives two directories deep (the Delta ``_delta_log``
    entry), so a non-recursive glob would miss it and raise FileNotFoundError.
    """
    pattern = f"{data_path}/readers/**/*0.json"
    reader = JsonReader(path=pattern)

    # Must not raise FileNotFoundError(matches=0); the nested file is discovered and read.
    df = reader.read()
    assert df.count() >= 1


def test_text_format_allows_mismatched_extension(csv_comma_file):
    """The text reader accepts any file, so a .csv path must not trigger an extension error."""
    reader = FileLoader(path=csv_comma_file, format="text")

    # Should not raise: reads the CSV as plain text lines.
    df = reader.read()
    assert df.count() == 4


def test_remote_and_multipath_skip_local_validation():
    """Remote URIs and comma-separated multi-paths must bypass local discovery checks."""
    assert _is_local_path("s3://bucket/data/file.csv") is False
    assert _is_local_path("s3a://bucket/data/file.csv") is False
    assert _is_local_path("hdfs://namenode/data/file.csv") is False
    assert _is_local_path("abfss://container@acct.dfs.core.windows.net/p") is False
    assert _is_local_path("dbfs:/mnt/data/file.csv") is False
    assert _is_local_path("file:/tmp/data/file.csv") is False
    assert _is_local_path("/data/a.csv,/data/b.csv") is False

    # Plain local paths (incl. Windows drive paths) are still inspected.
    assert _is_local_path("/data/local/file.csv") is True
    assert _is_local_path("relative/local/file.csv") is True
    assert _is_local_path("E:/work/data/file.csv") is True
