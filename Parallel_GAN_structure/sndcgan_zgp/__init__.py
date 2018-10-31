from .network import Network
import sndcgan_zgp.ops
import sndcgan_zgp.utils
import sndcgan_zgp.words

NAME = "SNDCGAN - Type B"
DETAIL = "Based on SNDCGAN ResNET(Parallel Net)."
VERSION = (2, 0, 0)
MODIFIED_DATE = "2018.10.30."
CHANGE_LOG = \
"""
Date        Changes
2018.08.29. New code structure established.
2018.10.30. Parallel model applied.
"""

DESCRIPTION = """----------------------------
| {name}
| detail: {detail}
| Version: {version[0]}.{version[1]}.{version[2]}
| Modified date: {date}
| Change log {change_log}
----------------------------""".format(name=NAME,
                                       detail=DETAIL,
                                       version=VERSION,
                                       date=MODIFIED_DATE,
                                       change_log=CHANGE_LOG.replace("\n", "\n|\t"))

print(DESCRIPTION)
