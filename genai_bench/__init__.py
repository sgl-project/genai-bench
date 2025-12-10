# Gevent monkey patching must be done before any other imports
# This is required for proper cooperative multitasking with Locust
# Without this, blocking I/O operations (like HTTP requests) will block
# the entire worker process, causing heartbeat timeouts
from gevent import monkey

monkey.patch_all()
