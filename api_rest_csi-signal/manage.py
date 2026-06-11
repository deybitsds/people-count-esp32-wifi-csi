#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys

# load the .env
from config.env import env, BASE_DIR
env.read_env(os.path.join(BASE_DIR, '.env'))

# 
def main():
    """Run administrative tasks."""
    # changed to work automatically
    # original path:  os.environ.setdefault('DJANGO_SETTINGS_MODULE','config.django.local')
    os.environ.setdefault(
        'DJANGO_SETTINGS_MODULE', 
        env('DJANGO_SETTINGS_MODULE', default='config.django.local'))

    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)

if __name__ == '__main__':
    main()
