"""
This module provides the RestApiReader class for interacting with RESTful APIs.

The RestApiReader class is designed to fetch data from RESTful APIs and store the response in a DataFrame. It supports
different transports, e.g. Paginated Http or Async HTTP. The main entry point is the `execute`
method, which performs transport.execute() call and provide data from the API calls.

For more details on how to use this class and its methods, refer to the class docstring.

"""

from typing import Any, List, Tuple, Union

from pydantic import Field, InstanceOf

# noinspection PyProtectedMember
from pyspark.sql.types import (
    AtomicType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from koheesio.asyncio.http import AsyncHttpGetStep
from koheesio.spark.readers import Reader
from koheesio.steps.http import HttpGetStep, PaginatedHttpGetStep

#: Schema of the nested struct column added to the output DataFrame when
#: ``include_request_metadata`` is enabled on :class:`RestApiReader`.
REQUEST_METADATA_SCHEMA = StructType(
    [
        StructField("request_url", StringType(), True),
        StructField("status_code", IntegerType(), True),
        StructField("page", IntegerType(), True),
        StructField("offset", IntegerType(), True),
    ]
)


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
    include_request_metadata : bool, optional, default=False
        When True, attach request metadata (request_url, status_code, page, offset) to the output
        DataFrame as a nested struct column. Note: status_code is unavailable for the async transport
        and page/offset are only populated for paginated requests; missing values are returned as null.
    metadata_column : str, optional, default="_request_metadata"
        Name of the nested struct column holding request metadata when include_request_metadata is
        True. Requires spark_schema to resolve to a StructType (StructType or DDL string).

    Attributes
    ----------
    transport : Union[InstanceOf[AsyncHttpGetStep], InstanceOf[HttpGetStep]]
        The HTTP transport step.
    spark_schema : Union[str, StructType, List[str], Tuple[str, ...], AtomicType]
        The pyspark schema of the response.

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

    """

    transport: Union[InstanceOf[AsyncHttpGetStep], InstanceOf[HttpGetStep]] = Field(
        ..., description="HTTP transport step", exclude=True
    )
    spark_schema: Union[str, StructType, List[str], Tuple[str, ...], AtomicType] = Field(
        ..., description="The pyspark schema of the response"
    )
    include_request_metadata: bool = Field(
        default=False,
        description=(
            "When True, attach request metadata (request_url, status_code, page, offset) to the "
            "output DataFrame as a nested struct column. Defaults to False, leaving the output "
            "unchanged."
        ),
    )
    metadata_column: str = Field(
        default="_request_metadata",
        description=(
            "Name of the nested struct column holding request metadata when "
            "include_request_metadata is True."
        ),
    )

    def execute(self) -> Reader.Output:
        """
        Executes the API call and stores the response in a DataFrame.

        When ``include_request_metadata`` is False (the default), the response is loaded exactly as
        before. When True, a nested struct column (named by ``metadata_column``) is appended with the
        request URL, HTTP status code and page number for each record.

        Returns
        -------
        Reader.Output
            The output of the reader, which includes the DataFrame.
        """
        raw_data = self.transport.execute()

        if self.include_request_metadata:
            self._set_output_with_metadata(raw_data)
        else:
            data = None
            if isinstance(raw_data, HttpGetStep.Output):
                data = raw_data.response_json
            elif isinstance(raw_data, AsyncHttpGetStep.Output):
                data = [d for d, _ in raw_data.responses_urls]  # type: ignore

            if data:
                self.output.df = self.spark.createDataFrame(data=data, schema=self.spark_schema)  # type: ignore

    def _set_output_with_metadata(self, raw_data: Union[HttpGetStep.Output, AsyncHttpGetStep.Output]) -> None:
        """Build the output DataFrame with a nested request-metadata struct column appended."""
        records = self._collect_with_metadata(raw_data)
        if not records:
            return

        base_schema = self._resolve_struct_schema()
        extended_schema = StructType(
            list(base_schema.fields) + [StructField(self.metadata_column, REQUEST_METADATA_SCHEMA, True)]
        )
        rows = [{**record, self.metadata_column: metadata} for record, metadata in records]
        self.output.df = self.spark.createDataFrame(data=rows, schema=extended_schema)

    def _resolve_struct_schema(self) -> StructType:
        """Resolve ``spark_schema`` to a StructType so the metadata column can be appended.

        Reuses the same schema parsing that ``createDataFrame`` relies on for DDL strings. Raises if
        the schema is not struct-like (e.g. a bare AtomicType or a list of column names), since
        appending a named metadata column is only meaningful for a struct schema.
        """
        if isinstance(self.spark_schema, StructType):
            return self.spark_schema
        if isinstance(self.spark_schema, str):
            return self.spark.createDataFrame([], schema=self.spark_schema).schema  # type: ignore[arg-type]
        raise ValueError(
            "include_request_metadata=True requires a StructType or DDL-string spark_schema; "
            f"got {type(self.spark_schema).__name__}."
        )

    def _collect_with_metadata(
        self, raw_data: Union[HttpGetStep.Output, AsyncHttpGetStep.Output]
    ) -> List[Tuple[dict, dict]]:
        """Pair each response record with its request metadata (request_url, status_code, page, offset)."""
        results: List[Tuple[dict, dict]] = []

        # Paginated transport carries real per-page metadata. Checked first because its Output is a
        # subclass of HttpGetStep.Output.
        if isinstance(self.transport, PaginatedHttpGetStep) and getattr(raw_data, "paginated_responses", None):
            for page_response in raw_data.paginated_responses:  # type: ignore[union-attr]
                metadata = {
                    "request_url": page_response.get("url"),
                    "status_code": page_response.get("status_code"),
                    "page": page_response.get("page"),
                    # offset is a single request-level config (the starting page), constant
                    # across pages; read from the transport since paginated_responses omits it.
                    "offset": self.transport.offset,
                }
                results.extend(self._expand_records(page_response.get("data"), metadata))
            return results

        if isinstance(raw_data, HttpGetStep.Output):
            response_url = raw_data.response_raw.url if raw_data.response_raw is not None else self.transport.url
            metadata = {
                "request_url": str(response_url) if response_url is not None else None,
                "status_code": raw_data.status_code,
                "page": None,
                "offset": None,
            }
            results.extend(self._expand_records(raw_data.response_json, metadata))
            return results

        if isinstance(raw_data, AsyncHttpGetStep.Output):
            for record_data, url in raw_data.responses_urls or []:  # type: ignore[union-attr]
                metadata = {"request_url": str(url), "status_code": None, "page": None, "offset": None}
                results.extend(self._expand_records(record_data, metadata))
            return results

        return results

    @staticmethod
    def _expand_records(data: Any, metadata: dict) -> List[Tuple[dict, dict]]:
        """Pair each record in ``data`` with the same metadata; lists fan out, scalars stay single."""
        if data is None:
            return []
        items = data if isinstance(data, list) else [data]
        return [(item, metadata) for item in items]
