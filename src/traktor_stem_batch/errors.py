class StemBatchError(RuntimeError):
    pass


class MissingDependencyError(StemBatchError):
    pass


class BackendError(StemBatchError):
    pass


class CollectionError(StemBatchError):
    pass
