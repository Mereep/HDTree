from .base import HDTreeBaseException
class HDTreeSplitException(HDTreeBaseException):
    def __init__(self, message, code: int):
        super().__init__(message, code=code)
