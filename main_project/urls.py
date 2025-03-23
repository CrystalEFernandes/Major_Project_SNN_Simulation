# urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('api/update_routing/', views.update_routing, name='update_routing'),
    path('api/update_status/', views.update_status, name='update_status'),
    path('api/get_network_status/', views.get_network_status_data, name='get_network_status_data'),
    path('api/snn_settings/', views.get_snn_settings, name='snn_settings'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('history/', views.historical_routing_view, name='historical_routing_view'),
    path('api/simulation_data/', views.receive_simulation_data, name='receive_simulation_data'),

    path('simulation-data/', views.simulation_data_point_list, name='simulation_data_list'),
    path('simulation-data/<int:pk>/', views.simulation_data_point_detail, name='simulation_data_detail'),

    path('simulation-log/', views.view_simulation_log, name='simulation_log'),

    path('snn-control-panel/', views.snn_control_panel_view, name='snn_control_panel'),

    path('simulation-report/<int:simulation_id>/', views.view_simulation_report, name='simulation_report'),

]