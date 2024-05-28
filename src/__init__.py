# the logger objects will be created in the __init__ file
from time import strftime

from .utils.logs import create_logger


def init_logger(date_suffix=None, **kwargs):
    date = strftime("%Y%m%d")
    time = strftime("%Hh%Mmin%Ss")
    logger, logger_file, path_logs, final_dest = create_logger(
        "main", date, time, date_suffix, **kwargs)
    return logger, date, time, path_logs, final_dest
