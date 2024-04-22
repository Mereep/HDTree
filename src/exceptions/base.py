class HDTreeBaseException(Exception):
    def __init__(self, message: str, code: int):
        self.code = code
        self.message = message

    def __str__(self):
        return self.message + " (Code: " + str(self.code) + ")"
