from aiohttp import ClientSession, TCPConnector
from aiohttp_retry import ExponentialRetry
from aioresponses import aioresponses
import pytest
import responses
from responses.registries import OrderedRegistry
from yarl import URL

from pyspark.sql.types import MapType, StringType, StructField, StructType

from koheesio.asyncio.http import AsyncHttpStep
from koheesio.spark.readers.rest_api import AsyncHttpGetStep, RestApiReader
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


@responses.activate
def test_metadata_disabled_by_default():
    """By default no metadata column is added and the output is unchanged."""
    responses.get(str(ASYNC_GET_ENDPOINT), json=[{"id": 1, "value": "a"}])

    transport = HttpGetStep(url=str(ASYNC_GET_ENDPOINT))
    task = RestApiReader(transport=transport, spark_schema="id: int, value: string")

    assert task.include_request_metadata is False

    task.execute()

    rows = [row.asDict() for row in task.output.df.collect()]
    assert rows == [{"id": 1, "value": "a"}]
    assert "_request_metadata" not in task.output.df.columns


@responses.activate(registry=OrderedRegistry)
def test_paginated_api_with_metadata():
    """Enabling metadata preserves per-page url/status/page alongside the response fields."""
    for i in range(1, 4):  # Mock 3 pages of data
        data = [{"id": j, "page": i, "value": f"data_{i}_{j}"} for j in range(1, 11)]  # 10 records per page
        responses.get(f"https://api.example.com/data?page={i}", json=data)

    transport = PaginatedHttpGetStep(url="https://api.example.com/data?page={page}", paginate=True, pages=3)
    task = RestApiReader(
        transport=transport,
        spark_schema="id: int, page:int, value: string",
        include_request_metadata=True,
    )

    task.execute()

    assert "_request_metadata" in task.output.df.columns

    rows = [row.asDict(recursive=True) for row in task.output.df.collect()]
    assert len(rows) == 30

    for row in rows:
        meta = row["_request_metadata"]
        # The response payload's own 'page' field is preserved and matches the metadata page,
        # proving the nested struct does not collide with response fields.
        assert meta["page"] == row["page"]
        assert meta["status_code"] == 200
        assert meta["request_url"] == f"https://api.example.com/data?page={row['page']}"

    for page in (1, 2, 3):
        assert sum(1 for row in rows if row["_request_metadata"]["page"] == page) == 10

    # The original (flattened) response data is still fully intact.
    expected_data = [{"id": j, "page": i, "value": f"data_{i}_{j}"} for i in range(1, 4) for j in range(1, 11)]
    actual_data = [{"id": row["id"], "page": row["page"], "value": row["value"]} for row in rows]
    assert actual_data == expected_data


@responses.activate
def test_rest_api_reader_with_metadata():
    """Single GET requests expose request_url and status_code; page is null."""
    responses.get(str(ASYNC_GET_ENDPOINT), json=[{"origin": "x", "url": str(ASYNC_GET_ENDPOINT)}])

    transport = HttpGetStep(url=str(ASYNC_GET_ENDPOINT))
    spark_schema = StructType(
        [
            StructField("origin", StringType(), True),
            StructField("url", StringType(), True),
        ]
    )
    task = RestApiReader(transport=transport, spark_schema=spark_schema, include_request_metadata=True)

    task.execute()

    rows = [row.asDict(recursive=True) for row in task.output.df.collect()]
    assert len(rows) == 1

    row = rows[0]
    # Response field 'url' is preserved and distinct from the metadata 'request_url'.
    assert row["url"] == str(ASYNC_GET_ENDPOINT)
    meta = row["_request_metadata"]
    assert meta["status_code"] == 200
    assert meta["request_url"] == str(ASYNC_GET_ENDPOINT)
    assert meta["page"] is None


@pytest.mark.asyncio
async def test_async_rest_api_reader_with_metadata(mock_aiohttp):
    """Async requests expose per-row request_url; status_code/page are null (not captured)."""
    mock_aiohttp.get(str(ASYNC_GET_ENDPOINT), status=200, repeat=True, payload={"url": str(ASYNC_GET_ENDPOINT)})

    transport = AsyncHttpGetStep(
        client_session=ClientSession(),
        url=[URL(ASYNC_GET_ENDPOINT), URL(ASYNC_GET_ENDPOINT)],
        retry_options=ExponentialRetry(),
        connector=TCPConnector(limit=10),
    )

    spark_schema = StructType([StructField("url", StringType(), True)])
    task = RestApiReader(transport=transport, spark_schema=spark_schema, include_request_metadata=True)

    task.execute()

    rows = [row.asDict(recursive=True) for row in task.output.df.collect()]
    assert len(rows) == 2

    for row in rows:
        assert row["url"] == f"{ASYNC_BASE_URL}/get"
        meta = row["_request_metadata"]
        assert meta["request_url"] == f"{ASYNC_BASE_URL}/get"
        assert meta["status_code"] is None
        assert meta["page"] is None
