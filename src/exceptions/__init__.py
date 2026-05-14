import sys

def error_message_detail(error, sys):
    _, _, exc_tb = sys.exc_info()
    if exc_tb is None:
        return str(error)
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occurred python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return error_message

class CustomException(Exception):

    def __init__(self, error_message, sys):
        """ param error_message: error message in string format """
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, sys)

    def __str__(self):
        return self.error_message
