"""
This module provides the RestApiReader class for interacting with RESTful APIs.

The RestApiReader class is designed to fetch data from RESTful APIs and store the response in a DataFrame. It supports
different transports, e.g. Paginated Http or Async HTTP. The main entry point is the `execute`
method, which performs transport.execute() call and provide data from the API calls.

For more details on how to use this class and its methods, refer to the class docstring.

"""

from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import Field, InstanceOf

# noinspection PyProtectedMember
from pyspark.sql.types import AtomicType, IntegerType, StringType, StructField, StructType

from koheesio.asyncio.http import AsyncHttpGetStep
from koheesio.spark.readers import Reader
from koheesio.steps.http import HttpGetStep, PaginatedHttpGetStep


# Default metadata schema attached to every output row when ``include_metadata=True``.
DEFAULT_METADATA_SCHEMA = StructType(
    [
        StructField("request_url", StringType(), True),
        StructField("status_code", IntegerType(), True),
        StructField("page", IntegerType(), True),
        StructField("offset", IntegerType(), True),
    ]
)

# Column name used for the metadata struct in the output DataFrame.
METADATA_COLUMN_NAME = "_metadata"


# noinspection HttpUrlsUsage
class RestApiReader(Reader):
    # noinspection HttpUrlsUsage
    """
    A reader class that executes an API call and stores the response in a DataFrame.

    Parameters
    ----------
    transport : Union[InstanceOf[AsyncHttpGetStep], InstanceOf[HttpGetStep]]
        The HTTP transport step.
    spark_schema : Union[str, StructType, List[str], Tuple[str, ...], AtomicType]
        The pyspark schema of the response.
    include_metadata : bool, optional, default=False
        When True, each row in the output DataFrame will include a ``_metadata`` struct column
        containing request-level information such as the request URL, HTTP status code, page
        number, and offset.  The struct schema can be customised via ``metadata_schema``.
    metadata_schema : Optional[StructType], optional, default=None
        Custom Spark schema for the ``_metadata`` column.  When *None* the
        ``DEFAULT_METADATA_SCHEMA`` is used (request_url, status_code, page, offset).

    Attributes
    ----------
    transport : Union[InstanceOf[AsyncHttpGetStep], InstanceOf[HttpGetStep]]
        The HTTP transport step.
    spark_schema : Union[str, StructType, List[str], Tuple[str, ...], AtomicType]
        The pyspark schema of the response.
    include_metadata : bool
        Whether metadata is included in the output DataFrame.
    metadata_schema : Optional[StructType]
        Schema of the metadata struct column.

    Returns
    -------
    Reader.Output
        The output of the reader, which includes the DataFrame.


    Examples
    --------
    Here are some examples of how to use this class:

    Example 1: Paginated Transport
    ```python
    import requests
    from urllib3 import Retry

    from koheesio.steps.http import HttpGetStep
    from koheesio.spark.readers.rest_api import RestApiReader

    session = requests.Session()
    retry_logic = Retry(total=max_retries, status_forcelist=[503])
    session.mount("https://", HTTPAdapter(max_retries=retry_logic))
    session.mount("http://", HTTPAdapter(max_retries=retry_logic))

    transport = PaginatedHttpGetStep(
        url="https://api.example.com/data?page={page}",
        paginate=True,
        pages=3,
        session=session,
    )
    task = RestApiReader(
        transport=transport,
        spark_schema="id: int, page:int, value: string",
    )
    task.execute()
    all_data = [row.asDict() for row in task.output.df.collect()]
    ```

    Example 2: Async Transport
    ```python
    from aiohttp import ClientSession, TCPConnector
    from aiohttp_retry import ExponentialRetry
    from yarl import URL

    from koheesio.steps.asyncio.http import AsyncHttpGetStep
    from koheesio.spark.readers.rest_api import RestApiReader

    session = ClientSession()
    urls = [URL("http://httpbin.org/get"), URL("http://httpbin.org/get")]
    retry_options = ExponentialRetry()
    connector = TCPConnector(limit=10)
    transport = AsyncHttpGetStep(
        client_session=session,
        url=urls,
        retry_options=retry_options,
        connector=connector,
    )

    task = RestApiReader(
        transport=transport,
        spark_schema="id: int, page:int, value: string",
    )
    task.execute()
    all_data = [row.asDict() for row in task.output.df.collect()]
    ```

    Example 3: With metadata
    ```python
    transport = PaginatedHttpGetStep(
        url="https://api.example.com/data?page={page}",
        paginate=True,
        pages=3,
    )
    task = RestApiReader(
        transport=transport,
        spark_schema="id: int, page:int, value: string",
        include_metadata=True,
    )
    task.execute()
    # Each row now has a ``_metadata`` struct with request_url, status_code, page, offset.
    ```

    """

    transport: Union[InstanceOf[AsyncHttpGetStep], InstanceOf[HttpGetStep]] = Field(
        ..., description="HTTP transport step", exclude=True
    )
    spark_schema: Union[str, StructType, List[str], Tuple[str, ...], AtomicType] = Field(
        ..., description="The pyspark schema of the response"
    )
    include_metadata: bool = Field(
        default=False,
        description=(
            "When True, each output row includes a _metadata struct column with request "
            "URL, HTTP status code, page number, and offset."
        ),
    )
    metadata_schema: Optional[StructType] = Field(
        default=None,
        description="Custom schema for the _metadata column. Defaults to DEFAULT_METADATA_SCHEMA.",
    )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _effective_metadata_schema(self) -> StructType:
        """Return the metadata schema to use (explicit or default)."""
        return self.metadata_schema or DEFAULT_METADATA_SCHEMA

    @staticmethod
    def _attach_metadata(
        data: List,
        metadata_per_record: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Zip *data* records with their corresponding metadata dicts and return a
        list of dicts where each dict is ``{**record, "_metadata": meta}``.

        If a record is not a dict (e.g. a primitive), it is wrapped as
        ``{"value": record}`` before the metadata key is added.
        """
        enriched: List[Dict[str, Any]] = []
        for record, meta in zip(data, metadata_per_record):
            row = dict(record) if isinstance(record, dict) else {"value": record}
            row[METADATA_COLUMN_NAME] = meta
            enriched.append(row)
        return enriched

    # ------------------------------------------------------------------
    # Transport-specific data / metadata extraction
    # ------------------------------------------------------------------

    def _extract_from_http_get(
        self, raw_data: HttpGetStep.Output
    ) -> Tuple[List, Optional[List[Dict[str, Any]]]]:
        """Extract data and (optionally) per-record metadata from a single HttpGetStep."""
        data = raw_data.response_json
        if not data or not self.include_metadata:
            return data, None  # type: ignore[return-value]

        meta = {
            "request_url": self.transport.url,  # type: ignore[union-attr]
            "status_code": raw_data.status_code,
            "page": None,
            "offset": None,
        }
        return data, [meta] * len(data)  # type: ignore[return-value]

    def _extract_from_paginated_http_get(
        self, raw_data: PaginatedHttpGetStep.Output
    ) -> Tuple[List, Optional[List[Dict[str, Any]]]]:
        """Extract data and (optionally) per-record metadata from a PaginatedHttpGetStep."""
        data = raw_data.response_json
        if not data or not self.include_metadata:
            return data, None  # type: ignore[return-value]

        pages_metadata = raw_data.pages_metadata or []
        metadata_per_record: List[Dict[str, Any]] = []

        # PaginatedHttpGetStep concatenates response lists per page.
        # We distribute records evenly across pages, placing any remainder on
        # the earlier pages.
        n_pages = len(pages_metadata)
        n_records = len(data)
        if n_pages > 0:
            base = n_records // n_pages
            remainder = n_records % n_pages
            for i, page_meta in enumerate(pages_metadata):
                count = base + (1 if i < remainder else 0)
                meta = {
                    "request_url": page_meta.get("request_url"),
                    "status_code": page_meta.get("status_code"),
                    "page": page_meta.get("page"),
                    "offset": page_meta.get("offset"),
                }
                metadata_per_record.extend([meta] * count)
        else:
            # Fallback: no page metadata available – attach empty metadata.
            empty_meta = {"request_url": None, "status_code": None, "page": None, "offset": None}
            metadata_per_record = [empty_meta] * n_records

        return data, metadata_per_record  # type: ignore[return-value]

    def _extract_from_async(
        self, raw_data: AsyncHttpGetStep.Output
    ) -> Tuple[List, Optional[List[Dict[str, Any]]]]:
        """Extract data and (optionally) per-record metadata from an AsyncHttpGetStep."""
        if not raw_data.responses_urls:
            return [], None

        data = [d for d, _ in raw_data.responses_urls]
        if not self.include_metadata:
            return data, None

        responses_metadata = raw_data.responses_metadata or []
        # Each async response may itself be a list or a single dict.  Align
        # metadata with records: one meta entry per record in each response.
        metadata_per_record: List[Dict[str, Any]] = []
        for idx, response_data in enumerate(data):
            meta_src = responses_metadata[idx] if idx < len(responses_metadata) else {}
            meta = {
                "request_url": meta_src.get("request_url"),
                "status_code": meta_src.get("status_code"),
                "page": None,
                "offset": None,
            }
            if isinstance(response_data, list):
                metadata_per_record.extend([meta] * len(response_data))
                # Flatten the data list as well so records align 1-to-1.
            else:
                metadata_per_record.append(meta)

        # Flatten *data* when individual responses are lists so that each row
        # corresponds to a single record.
        flat_data: List = []
        for item in data:
            if isinstance(item, list):
                flat_data.extend(item)
            else:
                flat_data.append(item)

        return flat_data, metadata_per_record

    # ------------------------------------------------------------------
    # execute
    # ------------------------------------------------------------------

    def execute(self) -> Reader.Output:
        """
        Executes the API call and stores the response in a DataFrame.

        When ``include_metadata`` is *True*, an additional ``_metadata`` struct
        column is appended to every row containing request-level information
        (URL, status code, page/offset).

        Returns
        -------
        Reader.Output
            The output of the reader, which includes the DataFrame.
        """
        raw_data = self.transport.execute()

        data: Optional[List] = None
        metadata_per_record: Optional[List[Dict[str, Any]]] = None

        if isinstance(self.transport, PaginatedHttpGetStep):
            data, metadata_per_record = self._extract_from_paginated_http_get(raw_data)
        elif isinstance(raw_data, HttpGetStep.Output):
            data, metadata_per_record = self._extract_from_http_get(raw_data)
        elif isinstance(raw_data, AsyncHttpGetStep.Output):
            data, metadata_per_record = self._extract_from_async(raw_data)

        if data is not None:
            if metadata_per_record is not None and self.include_metadata:
                enriched = self._attach_metadata(data, metadata_per_record)
                schema = self.spark_schema
                if isinstance(schema, str):
                    schema = f"{schema}, {METADATA_COLUMN_NAME}: {self._effective_metadata_schema().simpleString()}"
                elif isinstance(schema, StructType):
                    schema = StructType(
                        schema.fields + [StructField(METADATA_COLUMN_NAME, self._effective_metadata_schema(), True)]
                    )
                self.output.df = self.spark.createDataFrame(data=enriched, schema=schema)  # type: ignore
            else:
                self.output.df = self.spark.createDataFrame(data=data, schema=self.spark_schema)  # type: ignore
