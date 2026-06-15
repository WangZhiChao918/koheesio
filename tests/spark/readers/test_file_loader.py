import logging

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


# ---------------------------------------------------------------------------
# Path validation & error-message tests
# ---------------------------------------------------------------------------


class TestPathNotFound:
    """Non-existent paths must raise FileNotFoundError with contextual info."""

    def test_nonexistent_file(self):
        reader = CsvReader(path="/no/such/file_or_dir/missing.csv", header=True)
        with pytest.raises(FileNotFoundError) as exc_info:
            reader.read()

        msg = str(exc_info.value)
        assert "/no/such/file_or_dir/missing.csv" in msg
        assert "csv" in msg
        assert "does not exist" in msg

    def test_nonexistent_path_with_json_reader(self):
        reader = JsonReader(path="/tmp/definitely_not_here/data.json")
        with pytest.raises(FileNotFoundError) as exc_info:
            reader.read()

        msg = str(exc_info.value)
        assert "json" in msg
        assert "/tmp/definitely_not_here/data.json" in msg


class TestGlobNoMatch:
    """Glob patterns that match zero files must raise FileNotFoundError."""

    def test_glob_zero_matches(self):
        reader = CsvReader(path="/no/such/dir/*.csv", header=True)
        with pytest.raises(FileNotFoundError) as exc_info:
            reader.read()

        msg = str(exc_info.value)
        assert "No files matched" in msg
        assert "*.csv" in msg
        assert "csv" in msg

    def test_glob_question_mark_no_match(self):
        reader = JsonReader(path="/no/such/dir/data_?.json")
        with pytest.raises(FileNotFoundError) as exc_info:
            reader.read()

        msg = str(exc_info.value)
        assert "No files matched" in msg


class TestGlobWithMatches:
    """Glob patterns that match files should read normally."""

    def test_glob_reads_matched_csv_files(self, data_path):
        pattern = f"{data_path}/readers/csv_file/*.csv"
        schema = "string STRING, int INT, float FLOAT"
        reader = CsvReader(path=pattern, header=True, schema=schema)
        df = reader.read()
        assert df.count() >= 3  # both CSV files combined


class TestDirectoryPathNoExtensionCheck:
    """Directory paths (e.g. parquet partitions) must not trigger extension warnings."""

    def test_parquet_directory_no_warning(self, parquet_file, caplog):
        with caplog.at_level(logging.WARNING):
            reader = ParquetReader(path=parquet_file)
            df = reader.read()
            assert df.count() >= 3

        # No extension-mismatch warning should appear for directory paths
        assert "Extension mismatch" not in caplog.text


class TestExtensionMismatchWarning:
    """Reading a file whose extension does not match the format should log a warning."""

    def test_csv_reader_on_json_file_warns(self, json_file, caplog):
        with caplog.at_level(logging.WARNING):
            reader = CsvReader(path=json_file, header=True)
            # We only exercise the validation path — the actual Spark read may
            # succeed or fail depending on file content, but the warning must
            # have been emitted before Spark is invoked.
            try:
                reader.read()
            except Exception:
                pass  # Spark may fail to parse JSON as CSV, that's fine

        assert "Extension mismatch" in caplog.text
        assert "csv" in caplog.text
        assert json_file in caplog.text

    def test_json_reader_on_csv_file_warns(self, csv_comma_file, caplog):
        with caplog.at_level(logging.WARNING):
            reader = JsonReader(path=csv_comma_file)
            try:
                reader.read()
            except Exception:
                pass

        assert "Extension mismatch" in caplog.text
        assert "json" in caplog.text

    def test_parquet_reader_on_csv_file_warns(self, csv_comma_file, caplog):
        with caplog.at_level(logging.WARNING):
            reader = ParquetReader(path=csv_comma_file)
            try:
                reader.read()
            except Exception:
                pass

        assert "Extension mismatch" in caplog.text
        assert "parquet" in caplog.text


class TestNormalPathsUnaffected:
    """Existing happy-path reads must continue to work without any change."""

    def test_csv_file_read(self, csv_comma_file):
        schema = "string STRING, int INT, float FLOAT"
        reader = CsvReader(path=csv_comma_file, header=True, schema=schema)
        df = reader.read()
        assert df.count() == 3

    def test_json_file_read(self, json_file):
        reader = JsonReader(path=json_file)
        df = reader.read()
        assert df.count() == 3

    def test_parquet_dir_read(self, parquet_file):
        reader = ParquetReader(path=parquet_file)
        df = reader.read()
        assert df.count() >= 3

    def test_orc_dir_read(self, orc_file):
        reader = OrcReader(path=orc_file)
        df = reader.read()
        assert df.count() == 3
