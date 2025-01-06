from celery import Celery
from config import CELERY_BROKER_URL, CELERY_RESULT_BACKEND

app = Celery('spaces', broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)

app.conf.beat_schedule = {
    'check-if-live': {
        'task': 'monitor.check_if_live',
        'schedule': 60.0,
    },
}