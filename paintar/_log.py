import logging
import pkgutil
from typing import Union

import paintar


def logger_setup(log_level: Union[int, str] = logging.ERROR, modules_log_level: int = logging.ERROR):
    # sets format for the log
    logging.basicConfig(format="%(levelname)s|%(name)s|%(message)s", level=modules_log_level)

    # sets log level for local packages
    for _, modname, _ in pkgutil.walk_packages(path=paintar.__path__,
                                               prefix=paintar.__name__ + ".",
                                               onerror=lambda x: None):
        logging.getLogger(modname).setLevel(log_level)