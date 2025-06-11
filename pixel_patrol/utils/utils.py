import math


def format_bytes_to_human_readable(size_bytes: int) -> str:
    """Formats a size in bytes to a human-readable string (KB, MB, GB, etc.)."""
    if size_bytes is None:
        return "N/A"
    if size_bytes == 0:
        return "0 Bytes"
    size_name = ("Bytes", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])
