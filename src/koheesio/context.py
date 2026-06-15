"""
The Context module is a part of the Koheesio framework and is primarily used for managing the environment configuration
where a Task or Step runs. It helps in adapting the behavior of a Task/Step based on the environment it operates in,
thereby avoiding the repetition of configuration values across different tasks.

The Context class, which is a key component of this module, functions similarly to a dictionary but with additional
features. It supports operations like handling nested keys, recursive merging of contexts, and
serialization/deserialization to and from various formats like JSON, YAML, and TOML.

For a comprehensive guide on the usage, examples, and additional features of the Context class, please refer to the
[reference/concepts/context](../reference/concepts/context.md) section of the Koheesio documentation.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Union
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import re

import jsonpickle  # type: ignore[import-untyped]
import tomli
import yaml

__all__ = ["Context", "ContextSource"]

# Reserved key used to stash source-tracking metadata inside a Context's ``__dict__``.
# It is deliberately filtered out of ``to_dict()`` and ``__str__`` so that source tracking never
# affects normal dict-like access, iteration, length, merging, or (de)serialization.
_PROVENANCE_KEY = "__koheesio_provenance__"


@dataclass(frozen=True)
class ContextSource:
    """Describes where a Context value originated, for debugging multi-source configurations.

    Parameters
    ----------
    source:
        Human-readable origin of the value. For values loaded from a file this is the file path;
        for values loaded from a raw string it is a marker such as ``"<yaml string>"``.
    env:
        Optional environment label supplied by the caller (e.g. ``"dev"``, ``"prod"``), useful for
        distinguishing which environment overlay contributed a value.
    """

    source: str
    env: Optional[str] = None


class Context(Mapping):
    """
    The Context class is a key component of the Koheesio framework, designed to manage configuration data and shared
    variables across tasks and steps in your application. It behaves much like a dictionary, but with added
    functionalities.

    Key Features
    ------------
    - _Nested keys_: Supports accessing and adding nested keys similar to dictionary keys.
    - _Recursive merging_: Merges two Contexts together, with the incoming Context having priority.
    - _Serialization/Deserialization_: Easily created from a yaml, toml, or json file, or a dictionary, and can be
        converted back to a dictionary.
    - _Handling complex Python objects_: Uses jsonpickle for serialization and deserialization of complex Python objects
        to and from JSON.

    For a comprehensive guide on the usage, examples, and additional features of the Context class, please refer to the
    [reference/concepts/context](../reference/concepts/context.md) section of the Koheesio documentation.

    Methods
    -------
    add(key: str, value: Any) -> Context
        Add a key/value pair to the context.
    get(key: str, default: Any = None, safe: bool = True) -> Any`
        Get value of a given key.
    get_item(key: str, default: Any = None, safe: bool = True) -> Dict[str, Any]
        Acts just like `.get`, except that it returns the key also.
    contains(key: str) -> bool
        Check if the context contains a given key.
    merge(context: Context, recursive: bool = False) -> Context
        Merge this context with the context of another, where the incoming context has priority.
    to_dict() -> dict
        Returns all parameters of the context as a dict.
    from_dict(kwargs: dict) -> Context
        Creates Context object from the given dict.
    from_yaml(yaml_file: str | Path) -> Context
        Creates Context object from a given yaml file.
    get_source(key: str, default: Any = None) -> ContextSource | None
        Return the effective source of a top-level key (debug aid for multi-source configs).
    get_source_history(key: str, default: Any = None) -> list[ContextSource] | None
        Return the full override chain for a top-level key, oldest first.
    sources() -> Dict[str, ContextSource]
        Return a mapping of every tracked top-level key to its effective source.
    from_json(json_file: str | Path) -> Context
        Creates Context object from a given json file.

    Dunder methods
    --------------

    - _`_iter__()`: Allows for iteration across a Context.
    - `__len__()`: Returns the length of the Context.
    - `__getitem__(item)`: Makes class subscriptable.

    Inherited from Mapping
    ----------------------

    - `items()`: Returns all items of the Context.
    - `keys()`: Returns all keys of the Context.
    - `values()`: Returns all values of the Context.
    """

    def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        """Initializes the Context object with given arguments."""
        for arg in args:
            if isinstance(arg, dict):
                kwargs.update(arg)
            if isinstance(arg, Context):
                kwargs.update(arg.to_dict())

        if kwargs:
            for key, value in kwargs.items():
                self.__dict__[key] = self.process_value(value)

    def __str__(self) -> str:
        """Returns a string representation of the Context."""
        return str({k: v for k, v in self.__dict__.items() if k != _PROVENANCE_KEY})

    def __repr__(self) -> str:
        """Returns a string representation of the Context."""
        return self.__str__()

    def __iter__(self) -> Iterator[str]:
        """Allows for iteration across a Context"""
        return self.to_dict().__iter__()

    def __len__(self) -> int:
        """Returns the length of the Context"""
        return self.to_dict().__len__()

    def __getattr__(self, item: str) -> Any:
        try:
            return self.get(item, safe=False)
        except KeyError as e:
            raise AttributeError(item) from e

    def __getitem__(self, item: str) -> Any:
        """Makes class subscriptable"""
        return self.get(item, safe=False)

    @classmethod
    def _recursive_merge(cls, target_context: Context, merge_context: Context) -> Context:
        """
        Recursively merge two dictionaries to an arbitrary depth.

        Parameters
        ----------
        target_context: Context
            Koheesio Context to merge into
        merge_context: Context
            Koheesio Context that is being merged

        Returns
        -------
        Context

        Example
        --------
        ```python
        context_1 = {
            "key_1": "val_1",
            "key_2": {
                "sub_key_1": "sub_val_1",
                "sub_key_2": "sub_val_2",
            },
            "key_3": ["item_1"]
        }

        context_2 = {
            "key_2": {
                "sub_key_2": "sub_val_2.1",
                "sub_key_3": "sub_val_3",
            },
            "key_3": ["item_2"]
        }

        result = {
            "key_1": "val_1",
            "key_2": {
                "sub_key_1": "sub_val_1",
                "sub_key_2": "sub_val_2.1",
                "sub_key_3": "sub_val_3",
            }
            "key_3": ["item_1","item_2]
        }
        ```
        """
        for k, v in merge_context.items():
            if k in target_context and isinstance(target_context[k], Context) and isinstance(v, Context):
                cls._recursive_merge(target_context[k], merge_context[k])
            elif k in target_context and isinstance(target_context[k], list) and isinstance(v, list):
                # merge list items
                target_context[k].extend(merge_context[k])
            else:
                target_context.add(k, v)
        return target_context

    @classmethod
    def from_dict(cls, kwargs: dict) -> Context:
        """Creates Context object from the given dict

        Parameters
        ----------
        kwargs: dict

        Returns
        -------
        Context
        """
        return cls(kwargs)

    @classmethod
    def from_json(cls, json_file_or_str: Union[str, Path], env: Optional[str] = None) -> Context:
        """Creates Context object from a given json file

        Note: jsonpickle is used to serialize/deserialize the Context object. This is done to allow for objects to be
        stored in the Context object, which is not possible with the standard json library.

        Why jsonpickle?
        ---------------
        (from https://jsonpickle.github.io/)

        > Data serialized with python’s pickle (or cPickle or dill) is not easily readable outside of python. Using the
        json format, jsonpickle allows simple data types to be stored in a human-readable format, and more complex
        data types such as numpy arrays and pandas dataframes, to be machine-readable on any platform that supports
        json.

        Security
        --------
        (from https://jsonpickle.github.io/)

        > jsonpickle should be treated the same as the Python stdlib pickle module from a security perspective.

        ### ! Warning !
        > The jsonpickle module is not secure. Only unpickle data you trust.
        It is possible to construct malicious pickle data which will execute arbitrary code during unpickling.
        Never unpickle data that could have come from an untrusted source, or that could have been tampered with.
        Consider signing data with an HMAC if you need to ensure that it has not been tampered with.
        Safer deserialization approaches, such as reading JSON directly, may be more appropriate if you are processing
        untrusted data.

        Parameters
        ----------
        json_file_or_str : Union[str, Path]
            Pathlike string or Path that points to the json file or string containing json
        env : Optional[str], optional, default=None
            Optional environment label recorded as the source environment for debugging.

        Returns
        -------
        Context
        """
        json_str = json_file_or_str

        # check if json_str is pathlike
        if (json_file := Path(json_file_or_str)).exists():
            json_str = json_file.read_text(encoding="utf-8")
            source = str(json_file)
        else:
            source = "<json string>"

        json_dict = jsonpickle.loads(json_str)
        return cls.from_dict(json_dict)._record_sources(source=source, env=env)

    @classmethod
    def from_toml(cls, toml_file_or_str: Union[str, Path], env: Optional[str] = None) -> Context:
        """Creates Context object from a given toml file

        Parameters
        ----------
        toml_file_or_str: Union[str, Path]
            Pathlike string or Path that points to the toml file or string containing toml
        env : Optional[str], optional, default=None
            Optional environment label recorded as the source environment for debugging.

        Returns
        -------
        Context
        """

        # check if toml_str is pathlike
        if (toml_file := Path(toml_file_or_str)).exists():
            toml_str = toml_file.read_text(encoding="utf-8")
            source = str(toml_file)
        else:
            toml_str = str(toml_file_or_str)
            source = "<toml string>"

        toml_dict = tomli.loads(toml_str)
        return cls.from_dict(toml_dict)._record_sources(source=source, env=env)

    @classmethod
    def from_yaml(cls, yaml_file_or_str: str, env: Optional[str] = None) -> Context:
        """Creates Context object from a given yaml file

        Parameters
        ----------
        yaml_file_or_str: str or Path
            Pathlike string or Path that points to the yaml file, or string containing yaml
        env : Optional[str], optional, default=None
            Optional environment label recorded as the source environment for debugging.

        Returns
        -------
        Context
        """
        yaml_str = yaml_file_or_str

        # check if yaml_str is pathlike
        if (yaml_file := Path(yaml_file_or_str)).exists():
            yaml_str = yaml_file.read_text(encoding="utf-8")
            source = str(yaml_file)
        else:
            yaml_str = str(yaml_file_or_str)
            source = "<yaml string>"

        # Bandit: disable yaml.load warning
        yaml_dict = yaml.load(yaml_str, Loader=yaml.Loader)  # nosec B506: yaml_load

        return cls.from_dict(yaml_dict)._record_sources(source=source, env=env)

    def add(self, key: str, value: Any) -> Context:
        """Add a key/value pair to the context"""
        self.__dict__[key] = value
        return self

    def contains(self, key: str) -> bool:
        """Check if the context contains a given key

        Parameters
        ----------
        key: str

        Returns
        -------
        bool
        """
        try:
            self.get(key, safe=False)
            return True
        except KeyError:
            return False

    def get(self, key: str, default: Any = None, safe: bool = True) -> Any:
        """Get value of a given key

        The key can either be an actual key (top level) or the key of a nested value.
        Behaves a lot like a dict's `.get()` method otherwise.

        Parameters
        ----------
        key:
            Can be a real key, or can be a dotted notation of a nested key
        default:
            Default value to return
        safe:
            Toggles whether to fail or not when item cannot be found

        Returns
        -------
        Any
            Value of the requested item

        Example
        -------
        Example of a nested call:

        ```python
        context = Context({"a": {"b": "c", "d": "e"}, "f": "g"})
        context.get("a.b")
        ```

        Returns `c`
        """
        try:
            # in case key is directly available, or is written in dotted notation
            try:
                return self.__dict__[key]
            except KeyError:
                pass
            if "." in key:
                # handle nested keys
                nested_keys = key.split(".")
                value = self  # parent object
                for k in nested_keys:
                    value = value[k]  # iterate through nested values
                return value

            raise KeyError

        except (AttributeError, KeyError, TypeError) as e:
            if not safe:
                raise KeyError(f"requested key '{key}' does not exist in {self}") from e
            return default

    def get_item(self, key: str, default: Any = None, safe: bool = True) -> Dict[str, Any]:
        """Acts just like `.get`, except that it returns the key also

        Returns
        -------
        Dict[str, Any]
            key/value-pair of the requested item

        Example
        -------
        Example of a nested call:

        ```python
        context = Context({"a": {"b": "c", "d": "e"}, "f": "g"})
        context.get_item("a.b")
        ```

        Returns `{'a.b': 'c'}`
        """
        value = self.get(key, default, safe)
        return {key: value}

    def get_all(self) -> dict:
        """alias to to_dict()"""
        return self.to_dict()

    def get_source(self, key: str, default: Any = None) -> Any:
        """Return the *effective* source of a top-level key, for debugging multi-source configs.

        The effective source is the last entry in the key's override chain, i.e. the source that
        "won" when several configs were merged. Returns `default` when the key has no recorded
        source (for example, a Context that was never loaded from a file or labelled with `env`).

        Note: source tracking is recorded per top-level key.

        Parameters
        ----------
        key: str
            Top-level key to look up.
        default: Any
            Value to return when no source was recorded for `key`.

        Returns
        -------
        ContextSource or default
        """
        history = self._provenance().get(key)
        return history[-1] if history else default

    def get_source_history(self, key: str, default: Any = None) -> Any:
        """Return the full override chain for a top-level key, oldest first.

        Each element is a `ContextSource`. The first element is the source that originally set the
        key; the last element is the effective (winning) source. Returns `default` when the key has
        no recorded source.

        Parameters
        ----------
        key: str
            Top-level key to look up.
        default: Any
            Value to return when no source was recorded for `key`.

        Returns
        -------
        list[ContextSource] or default
        """
        history = self._provenance().get(key)
        return list(history) if history else default

    def sources(self) -> Dict[str, ContextSource]:
        """Return a mapping of every tracked top-level key to its effective source.

        Returns an empty dict when no source information has been recorded. Handy for dumping a
        full "where did each value come from" overview when debugging a merged configuration.

        Returns
        -------
        Dict[str, ContextSource]
        """
        return {key: history[-1] for key, history in self._provenance().items() if history}

    def _provenance(self) -> Dict[str, List[ContextSource]]:
        """Internal: return the source-tracking metadata, or an empty dict when none is recorded."""
        return self.__dict__.get(_PROVENANCE_KEY, {})

    def _record_sources(self, source: str, env: Optional[str] = None) -> Context:
        """Internal: stamp every current top-level key with the given source. Returns `self`."""
        provenance = {key: [ContextSource(source=source, env=env)] for key in self.keys()}
        if provenance:
            self.__dict__[_PROVENANCE_KEY] = provenance
        return self

    def _merge_provenance(
        self, left: Dict[str, List[ContextSource]], right: Dict[str, List[ContextSource]]
    ) -> Context:
        """Internal: combine two provenance maps onto `self` in override order. Returns `self`.

        For each key present in `self`, the chain from `left` (lower priority) is placed before the
        chain from `right` (higher priority / incoming), so the resulting order reflects how the
        value was overridden across sources.
        """
        combined: Dict[str, List[ContextSource]] = {}
        for key in self.keys():
            chain = [*left.get(key, []), *right.get(key, [])]
            if chain:
                combined[key] = chain
        if combined:
            self.__dict__[_PROVENANCE_KEY] = combined
        return self

    def merge(self, context: Context, recursive: bool = False) -> Context:
        """Merge this context with the context of another, where the incoming context has priority.

        Parameters
        ----------
        context: Context
            Another Context class
        recursive: bool
            Recursively merge two dictionaries to an arbitrary depth

        Returns
        -------
        Context
            updated context
        """
        # Snapshot source provenance from both sides before merging the data, so the override order
        # can be recorded (this context first, the incoming context last == highest priority).
        self_provenance = self._provenance()
        incoming_provenance = context._provenance()

        if recursive:
            merged = Context.from_dict(self._recursive_merge(target_context=self, merge_context=context).to_dict())
        else:
            # just merge on the top level keys
            merged = Context.from_dict({**self.to_dict(), **context.to_dict()})

        return merged._merge_provenance(self_provenance, incoming_provenance)

    def process_value(self, value: Any) -> Any:
        """Processes the given value, converting dictionaries to Context objects as needed."""
        if isinstance(value, dict):
            return self.from_dict(value)

        if isinstance(value, (list, set)):
            return [self.from_dict(v) if isinstance(v, dict) else v for v in value]

        return value

    def to_dict(self) -> Dict[str, Any]:
        """Returns all parameters of the context as a dict

        Returns
        -------
        dict
            containing all parameters of the context
        """
        result = {}

        for key, value in self.__dict__.items():
            if key == _PROVENANCE_KEY:
                # source-tracking metadata is internal and must never surface as data
                continue
            if isinstance(value, Context):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [e.to_dict() if isinstance(e, Context) else e for e in value]  # type: ignore[assignment]
            else:
                result[key] = value

        return result

    def to_json(self, pretty: bool = False) -> str:
        """Returns all parameters of the context as a json string

        Note: jsonpickle is used to serialize/deserialize the Context object. This is done to allow for objects to be
        stored in the Context object, which is not possible with the standard json library.

        Why jsonpickle?
        ---------------
        (from https://jsonpickle.github.io/)

        > Data serialized with python's pickle (or cPickle or dill) is not easily readable outside of python. Using the
        json format, jsonpickle allows simple data types to be stored in a human-readable format, and more complex
        data types such as numpy arrays and pandas dataframes, to be machine-readable on any platform that supports
        json.

        Parameters
        ----------
        pretty : bool, optional, default=False
            Toggles whether to return a pretty json string or not

        Returns
        -------
        str
            containing all parameters of the context
        """
        d = self.to_dict()
        return jsonpickle.dumps(d, indent=4) if pretty else jsonpickle.dumps(d)

    def to_yaml(self, clean: bool = False) -> str:
        """Returns all parameters of the context as a yaml string

        Parameters
        ----------
        clean: bool
            Toggles whether to remove `!!python/object:...` from yaml or not.
            Default: False

        Returns
        -------
        str
            containing all parameters of the context
        """
        # sort_keys=False to preserve order of keys
        yaml_str = yaml.dump(self.to_dict(), sort_keys=False)

        # remove `!!python/object:...` from yaml
        if clean:
            remove_pattern = re.compile(r"!!python/object:.*?\n")
            yaml_str = re.sub(remove_pattern, "\n", yaml_str)

        return yaml_str
