from aiohttp import ClientSession, TCPConnector
from aiohttp_retry import ExponentialRetry
from aioresponses import aioresponses
import pytest
import responses
from responses.registries import OrderedRegistry
from yarl import URL

from pyspark.sql.types import IntegerType, MapType, StringType, StructField, StructType

from koheesio.asyncio.http import AsyncHttpStep
from koheesio.spark.readers.rest_api import (
    DEFAULT_METADATA_SCHEMA,
    METADATA_COLUMN_NAME,
    AsyncHttpGetStep,
    RestApiReader,
)
from koheesio.steps.http import HttpGetStep, PaginatedHttpGetStep

ASYNC_BASE_URL = "https://42.koheesio.test"
ASYNC_GET_ENDPOINT = URL(f"{ASYNC_BASE_URL}/get")

pytestmark = pytest.mark.spark


@pytest.fixture(scope="function", name="mock_aiohttp")
def mock_aiohttp():
    with aioresponses() as m:
        yield m


@responses.activate(registry=OrderedRegistry)
def test_paginated_api():
    for i in range(1, 4):  # Mock 3 pages of data
        data = [{"id": j, "page": i, "value": f"data_{i}_{j}"} for j in range(1, 11)]  # 10 records per page
        responses.get(f"https://api.example.com/data?page={i}", json=data)

    # Test that the paginated API returns all the data
    transport = PaginatedHttpGetStep(url="https://api.example.com/data?page={page}", paginate=True, pages=3)
    task = RestApiReader(transport=transport, spark_schema="id: int, page:int, value: string")

    assert isinstance(task.transport, PaginatedHttpGetStep)

    task.execute()

    # Convert the DataFrame to a list of dictionaries
    all_data = [row.asDict() for row in task.output.df.collect()]
    expected_data = [{"id": j, "page": i, "value": f"data_{i}_{j}"} for i in range(1, 4) for j in range(1, 11)]

    assert all_data == expected_data


@pytest.mark.asyncio
async def test_async_rest_api_reader(mock_aiohttp):
    """
    Testing the AsyncHttpStep class.
    """
    mock_aiohttp.get(str(ASYNC_GET_ENDPOINT), status=200, repeat=True, payload={"url": str(ASYNC_GET_ENDPOINT)})

    transport = AsyncHttpGetStep(
        client_session=ClientSession(),
        url=[URL(ASYNC_GET_ENDPOINT), URL(ASYNC_GET_ENDPOINT)],
        retry_options=ExponentialRetry(),
        connector=TCPConnector(limit=10),
        headers={"Content-Type": "application/json", "X-type": "Koheesio RestApiReader Test"},
    )

    spark_schema = StructType(
        [
            StructField("origin", StringType(), True),
            StructField("url", StringType(), True),
        ]
    )
    task = RestApiReader(transport=transport, spark_schema=spark_schema)

    assert isinstance(task.transport, AsyncHttpStep)

    task.execute()

    rows = [row.asDict() for row in task.output.df.collect()]
    all_data = [row["url"] for row in rows]

    # Assert the responses_urls
    assert len(all_data) == 2
    assert all_data == [f"{ASYNC_BASE_URL}/get"] * 2


@responses.activate
def test_rest_api_reader(mock_aiohttp):
    """
    Testing the AsyncHttpStep class.
    """

    def request_callback(request):
        import json

        body = [
            {
                "headers": dict(request.headers),
                "url": str(ASYNC_GET_ENDPOINT),
            }
        ]
        return (200, request.headers, json.dumps(body))

    responses.add_callback(
        responses.GET,
        str(ASYNC_GET_ENDPOINT),
        callback=request_callback,
        content_type="application/json",
    )

    transport = HttpGetStep(
        url=str(ASYNC_GET_ENDPOINT),
        headers={"Content-Type": "application/json", "X-Type": "Koheesio RestApiReader Test"},
    )

    spark_schema = StructType(
        [
            StructField(
                "headers",
                StructType(
                    [
                        StructField("Accept", StringType(), True),
                        StructField("Accept-Encoding", StringType(), True),
                        StructField("Connection", StringType(), True),
                        StructField("Content-Type", StringType(), True),
                        StructField("User-Agent", StringType(), True),
                        StructField("X-Type", StringType(), True),
                    ]
                ),
                True,
            ),
            StructField("url", StringType(), True),
        ]
    )

    task = RestApiReader(transport=transport, spark_schema=spark_schema)

    assert isinstance(task.transport, HttpGetStep)

    task.execute()

    rows = [row.asDict() for row in task.output.df.collect()]
    all_data = [{row["url"]: row.get("headers", {}).asDict()["X-Type"]} for row in rows]

    # Assert the responses_urls
    assert len(all_data) == 1
    assert all_data == [{f"{ASYNC_BASE_URL}/get": "Koheesio RestApiReader Test"}]


# ===========================================================================
# Metadata feature tests
# ===========================================================================


@responses.activate(registry=OrderedRegistry)
def test_paginated_api_default_no_metadata():
    """Default behaviour: no _metadata column is added to the output DataFrame."""
    for i in range(1, 4):
        data = [{"id": j, "page": i, "value": f"data_{i}_{j}"} for j in range(1, 11)]
        responses.get(f"https://api.example.com/data?page={i}", json=data)

    transport = PaginatedHttpGetStep(url="https://api.example.com/data?page={page}", paginate=True, pages=3)
    task = RestApiReader(transport=transport, spark_schema="id: int, page:int, value: string")
    task.execute()

    # Verify backward-compatible data
    all_data = [row.asDict() for row in task.output.df.collect()]
    expected_data = [{"id": j, "page": i, "value": f"data_{i}_{j}"} for i in range(1, 4) for j in range(1, 11)]
    assert all_data == expected_data

    # No _metadata column should exist
    assert METADATA_COLUMN_NAME not in task.output.df.columns


@responses.activate(registry=OrderedRegistry)
def test_paginated_api_with_metadata():
    """Paginated transport with include_metadata=True exposes per-page metadata."""
    for i in range(1, 4):
        data = [{"id": j, "page": i, "value": f"data_{i}_{j}"} for j in range(1, 11)]
        responses.get(f"https://api.example.com/data?page={i}", json=data)

    transport = PaginatedHttpGetStep(url="https://api.example.com/data?page={page}", paginate=True, pages=3)
    task = RestApiReader(
        transport=transport,
        spark_schema="id: int, page:int, value: string",
        include_metadata=True,
    )
    task.execute()

    # _metadata column should be present
    assert METADATA_COLUMN_NAME in task.output.df.columns

    rows = [row.asDict() for row in task.output.df.collect()]
    assert len(rows) == 30  # 3 pages * 10 records

    # Every row carries a _metadata struct
    for row in rows:
        assert METADATA_COLUMN_NAME in row
        meta = row[METADATA_COLUMN_NAME]
        assert meta["status_code"] == 200
        assert meta["request_url"] is not None

    # Verify per-page metadata alignment (10 records per page)
    page_numbers = [row[METADATA_COLUMN_NAME]["page"] for row in rows]
    assert page_numbers[:10] == [1] * 10
    assert page_numbers[10:20] == [2] * 10
    assert page_numbers[20:30] == [3] * 10

    # Request URLs should reflect the page parameter
    request_urls = [row[METADATA_COLUMN_NAME]["request_url"] for row in rows]
    assert request_urls[0] == "https://api.example.com/data?page=1"
    assert request_urls[15] == "https://api.example.com/data?page=2"
    assert request_urls[25] == "https://api.example.com/data?page=3"


@responses.activate
def test_rest_api_reader_with_metadata():
    """Single HttpGetStep with include_metadata=True exposes request metadata."""

    def request_callback(request):
        import json

        body = [{"key": "value1"}, {"key": "value2"}]
        return (200, request.headers, json.dumps(body))

    responses.add_callback(
        responses.GET,
        str(ASYNC_GET_ENDPOINT),
        callback=request_callback,
        content_type="application/json",
    )

    transport = HttpGetStep(
        url=str(ASYNC_GET_ENDPOINT),
        headers={"Content-Type": "application/json"},
    )
    spark_schema = StructType([StructField("key", StringType(), True)])

    task = RestApiReader(transport=transport, spark_schema=spark_schema, include_metadata=True)
    task.execute()

    assert METADATA_COLUMN_NAME in task.output.df.columns

    rows = [row.asDict() for row in task.output.df.collect()]
    assert len(rows) == 2

    for row in rows:
        meta = row[METADATA_COLUMN_NAME]
        assert meta["status_code"] == 200
        assert meta["request_url"] == str(ASYNC_GET_ENDPOINT)
        # page/offset are None for non-paginated requests
        assert meta["page"] is None
        assert meta["offset"] is None


@pytest.mark.asyncio
async def test_async_rest_api_reader_with_metadata(mock_aiohttp):
    """Async transport with include_metadata=True exposes per-response metadata."""
    mock_aiohttp.get(
        str(ASYNC_GET_ENDPOINT),
        status=200,
        repeat=True,
        payload={"url": str(ASYNC_GET_ENDPOINT)},
    )

    transport = AsyncHttpGetStep(
        client_session=ClientSession(),
        url=[URL(ASYNC_GET_ENDPOINT), URL(ASYNC_GET_ENDPOINT)],
        retry_options=ExponentialRetry(),
        connector=TCPConnector(limit=10),
        headers={"Content-Type": "application/json"},
    )

    spark_schema = StructType(
        [
            StructField("url", StringType(), True),
        ]
    )

    task = RestApiReader(transport=transport, spark_schema=spark_schema, include_metadata=True)
    task.execute()

    assert METADATA_COLUMN_NAME in task.output.df.columns

    rows = [row.asDict() for row in task.output.df.collect()]
    assert len(rows) == 2

    for row in rows:
        meta = row[METADATA_COLUMN_NAME]
        assert meta["status_code"] == 200
        assert meta["request_url"] == str(ASYNC_GET_ENDPOINT)


@pytest.mark.asyncio
async def test_async_rest_api_reader_default_no_metadata(mock_aiohttp):
    """Async transport default behaviour: no _metadata column."""
    mock_aiohttp.get(
        str(ASYNC_GET_ENDPOINT),
        status=200,
        repeat=True,
        payload={"url": str(ASYNC_GET_ENDPOINT)},
    )

    transport = AsyncHttpGetStep(
        client_session=ClientSession(),
        url=[URL(ASYNC_GET_ENDPOINT)],
        retry_options=ExponentialRetry(),
        connector=TCPConnector(limit=10),
    )

    spark_schema = StructType([StructField("url", StringType(), True)])
    task = RestApiReader(transport=transport, spark_schema=spark_schema)
    task.execute()

    assert METADATA_COLUMN_NAME not in task.output.df.columns
    rows = [row.asDict() for row in task.output.df.collect()]
    assert len(rows) == 1
    assert rows[0]["url"] == str(ASYNC_GET_ENDPOINT)
