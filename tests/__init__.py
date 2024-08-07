# work-around for https://github.com/locustio/locust/issues/2250
from gevent import monkey

monkey.patch_all()
