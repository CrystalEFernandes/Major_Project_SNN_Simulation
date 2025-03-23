from django import template
import json

register = template.Library()

@register.filter(name='json_loads')
def json_loads_filter(value):
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {} # Or handle error case as needed, e.g., return None, log error