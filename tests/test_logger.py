from memory.log_util import log_exception

def test_log_exception():
    try:
        raise ValueError("This is a test")
    except ValueError:
        log_exception()