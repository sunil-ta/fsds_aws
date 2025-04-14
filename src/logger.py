import logging
import os


class Logger:
    def __init__(self, filename, message, filemode):
        self.filename = filename
        self.message = message
        self.filemode = filemode

    def logging(self):
        log_dir = os.path.dirname(self.filename)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Ensure file exists
        if not os.path.exists(self.filename):
            open(self.filename, "w").close()

        logging.basicConfig(
            filename=self.filename,
            level=logging.DEBUG,
            format="%(asctime)s %(message)s",
            filemode=self.filemode,
        )
        logger = logging.getLogger()
        logger.debug(self.message)
