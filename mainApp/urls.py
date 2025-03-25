from django.contrib import admin
from django.urls import path,include
from . import views
urlpatterns = [
   
    path('compare/',views.routing_comparison_view,name='index'),
]
