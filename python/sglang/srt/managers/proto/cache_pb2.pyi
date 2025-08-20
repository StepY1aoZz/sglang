from google.protobuf import empty_pb2 as _empty_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class GetStartRequest(_message.Message):
    __slots__ = ("keys",)
    KEYS_FIELD_NUMBER: _ClassVar[int]
    keys: _containers.RepeatedScalarFieldContainer[bytes]
    def __init__(self, keys: _Optional[_Iterable[bytes]] = ...) -> None: ...

class Handle(_message.Message):
    __slots__ = ("key", "offset", "len")
    KEY_FIELD_NUMBER: _ClassVar[int]
    OFFSET_FIELD_NUMBER: _ClassVar[int]
    LEN_FIELD_NUMBER: _ClassVar[int]
    key: bytes
    offset: int
    len: int
    def __init__(self, key: _Optional[bytes] = ..., offset: _Optional[int] = ..., len: _Optional[int] = ...) -> None: ...

class GetStartReply(_message.Message):
    __slots__ = ("batchID", "handles")
    BATCHID_FIELD_NUMBER: _ClassVar[int]
    HANDLES_FIELD_NUMBER: _ClassVar[int]
    batchID: int
    handles: _containers.RepeatedCompositeFieldContainer[Handle]
    def __init__(self, batchID: _Optional[int] = ..., handles: _Optional[_Iterable[_Union[Handle, _Mapping]]] = ...) -> None: ...

class GetEndRequest(_message.Message):
    __slots__ = ("batchID", "revoked")
    BATCHID_FIELD_NUMBER: _ClassVar[int]
    REVOKED_FIELD_NUMBER: _ClassVar[int]
    batchID: int
    revoked: bool
    def __init__(self, batchID: _Optional[int] = ..., revoked: bool = ...) -> None: ...

class PutStartRequest(_message.Message):
    __slots__ = ("keys", "lens")
    KEYS_FIELD_NUMBER: _ClassVar[int]
    LENS_FIELD_NUMBER: _ClassVar[int]
    keys: _containers.RepeatedScalarFieldContainer[bytes]
    lens: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, keys: _Optional[_Iterable[bytes]] = ..., lens: _Optional[_Iterable[int]] = ...) -> None: ...

class PutStartReply(_message.Message):
    __slots__ = ("batchID", "offsets", "exists")
    BATCHID_FIELD_NUMBER: _ClassVar[int]
    OFFSETS_FIELD_NUMBER: _ClassVar[int]
    EXISTS_FIELD_NUMBER: _ClassVar[int]
    batchID: int
    offsets: _containers.RepeatedScalarFieldContainer[int]
    exists: _containers.RepeatedScalarFieldContainer[bool]
    def __init__(self, batchID: _Optional[int] = ..., offsets: _Optional[_Iterable[int]] = ..., exists: _Optional[_Iterable[bool]] = ...) -> None: ...

class PutEndRequest(_message.Message):
    __slots__ = ("batchID",)
    BATCHID_FIELD_NUMBER: _ClassVar[int]
    batchID: int
    def __init__(self, batchID: _Optional[int] = ...) -> None: ...
