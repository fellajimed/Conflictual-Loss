import logging
from pathlib import Path
from time import strftime


def create_logger(logger_name="main", date=strftime("%Y%m%d"),
                  time=strftime("%Hh%Mmin%Ss"), date_suffix=None,
                  path_logs=None, **kwargs):
    '''
    create the logger objects based on the date and the time
    2 loggers will be created:
        * a logger with the file handler and console handler
        * a logger with only the file handler
    '''
    # remove existing loggers
    logger = logging.getLogger(f"{logger_name}_all")
    logger.handlers.clear()
    logger_file = logging.getLogger(f"{logger_name}_file")
    logger_file.handlers.clear()
    del logger, logger_file

    if isinstance(date_suffix, str):
        date += f'_{date_suffix}'

    final_dest = None

    # path to the folder of the logs
    if path_logs is not None:
        path_logs = Path(path_logs).resolve().absolute()
        if ('team_storage' in str(path_logs)
                or 'orpailleur@talc-data' in str(path_logs)):
            final_dest = path_logs / date
            path_logs = Path('/tmp').resolve().absolute()

        path_logs = Path(path_logs).resolve().absolute() / date / time
    else:
        path_logs = (Path(__file__).resolve().parents[2].absolute()
                     / "logs" / date / time)

    # create the folder if it does not exist
    path_logs.mkdir(parents=True, exist_ok=True)

    # loggers
    logger = logging.getLogger(f"{logger_name}_all")
    logger.setLevel(level=logging.INFO)
    logger_file = logging.getLogger(f"{logger_name}_file")
    logger_file.setLevel(level=logging.INFO)

    formatter_str = "[%(levelname)s] [%(asctime)s] %(message)s"
    formatter = logging.Formatter(formatter_str)

    # file handler
    file_handler = logging.FileHandler(filename=path_logs/"logs.log",
                                       encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger_file.addHandler(file_handler)

    # console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger, logger_file, path_logs, final_dest
