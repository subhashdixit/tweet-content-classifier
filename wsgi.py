"""Gunicorn and local Flask entry point."""

from tweet_classifier.web import create_app

app = create_app()
