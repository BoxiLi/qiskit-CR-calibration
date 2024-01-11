import logging


def setup_logger(filename=None, level=logging.NOTSET, stdout=True):
    """
    This initializes a logger with the name ``qiskit_utilities``.
    By default, any logger created via ``logging.getLogger``under the
    name "qiskit_utilities.*" inherits the settings of this logger.

    Notice that we cannot set the global ``logging.basicConfig`` with ``logging.INFO``, because it will trigger lots of internal logging of qiskit.
    Hence we have to set this up in this way for a logger under the domain name qiskit_utilities.

    Caveat: Logging is tread-safe, but logging to the same file across multiple processes is not supported.
    """
    logging.basicConfig(level=logging.WARNING, force=True)
    logger = logging.getLogger("qiskit_utilities")
    if logger.hasHandlers():
        logger.handlers.clear()
    # Prevent the propagation of the log message to the root logger, which will duplicate the console log message.
    logger.propagate = False
    if filename is not None:
        formatter = logging.Formatter(
            "%(asctime)s, %(msecs)d %(threadName)s %(name)s\n%(levelname)s %(message)s"
        )
        file_handler = logging.FileHandler(filename, mode="a")
        file_handler.setFormatter(formatter)
        # file_handler.setLevel(level)
        logger.addHandler(file_handler)
    if stdout:
        formatter = logging.Formatter(
            "%(asctime)s, %(threadName)s \n%(levelname)s %(message)s"
        )
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        # stream_handler.setLevel(level)
        logger.addHandler(stream_handler)
    logger.setLevel(level)
    logger.info("Logger initialized successfully.")
