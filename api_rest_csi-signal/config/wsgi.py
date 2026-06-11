"""
WSGI config for DSCont_chatbot project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.2/howto/deployment/wsgi/
"""

import os
from config.env import env

from django.core.wsgi import get_wsgi_application

# changed to work automatically
# original path:  os.environ.setdefault('DJANGO_SETTINGS_MODULE','config.django.local')
os.environ.setdefault(
        'DJANGO_SETTINGS_MODULE', 
        env('DJANGO_SETTINGS_MODULE', default='config.django.local'))

application = get_wsgi_application()
