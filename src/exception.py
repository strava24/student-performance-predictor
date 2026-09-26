# Custom implementation to handle errors and exceptions

import sys # sys helps us to access the python interpreter level 

def error_message_detail(error, error_detail:sys):
    _, _, exc_tb = error_detail.exc_info() # ignorin the first 2 retuns 
    file_name = exc_tb.tb_frame.f_code.co_filename
    return f"Error Occured in [{file_name}], line number [{exc_tb.tb_lineno}], error message [{str(error)}]"

class CustomException(Exception):

    # Constructor
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    # toString representation
    def __str__(self):
        return self.error_message
