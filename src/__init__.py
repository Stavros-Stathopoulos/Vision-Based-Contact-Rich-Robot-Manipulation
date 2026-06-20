import os
import sys
import logging

# 1. Κλειδώνουμε το logging της Python να ΜΗΝ τυπώνει τίποτα (εκεί σταματάνε όλα τα custom warnings)
logging.disable(logging.WARNING)

# 2. Φιμώνουμε και τα standard warnings της Python/C-extensions
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"