from django.contrib import admin
from .models import (
    SNNSettings,
    NetworkStatus,
    RoutingPath,
    NeuronMetrics,
    NetworkGraphSnapshot,
    PathHistory,
)

from .models import SimulationDataPoint

@admin.register(SimulationDataPoint)
class SimulationDataPointAdmin(admin.ModelAdmin):
    list_display = ('timestamp', 'simulation_time_ms')  # Customize display in admin list view
    search_fields = ('timestamp',)  # Add search field for timestamp
    list_filter = ('timestamp',)  # Add filter by timestamp
    readonly_fields = ('firing_rates_json', 'latencies_ms_json', 'energy_levels_json', 'congestion_levels_json') # Make these fields read-only in the admin

    # Optional: If you want to display the related RoutingPath in the list view (be mindful of performance if you have many related objects)
    def routing_path_display(self, obj):
        return obj.routing_path.neighbor_scores_history_json if obj.routing_path else None  # Assuming your RoutingPath model has a 'name' field
    routing_path_display.short_description = 'Routing Path' # Set a more readable name for the column
    list_display = ('timestamp', 'simulation_time_ms', 'routing_path_display') # Add it to the list display
# Register your models here.

@admin.register(SNNSettings)
class SNNSettingsAdmin(admin.ModelAdmin):
    list_display = (
        'learning_rate',
        'synaptic_weight',
        'firing_threshold',
        'reset_potential',
        'tau_pre',
        'tau_post',
        'simulation_time',
        'created_at',
    )
    list_filter = ('created_at',)
    search_fields = ('learning_rate', 'synaptic_weight')


@admin.register(NetworkStatus)
class NetworkStatusAdmin(admin.ModelAdmin):
    list_display = (
        'timestamp',
        'total_nodes',
        'active_nodes',
        'total_packets',
        'avg_latency',
        'avg_energy',
    )
    list_filter = ('timestamp',)
    search_fields = ('total_nodes', 'active_nodes')


@admin.register(RoutingPath)
class RoutingPathAdmin(admin.ModelAdmin):
    list_display = (
        'timestamp',
        'source_node',
        'destination_node',
        'stdp_path',
        'stdp_path_score',
        'neighbor_scores_history_json',
    )
    list_filter = ('timestamp', 'source_node', 'destination_node')
    search_fields = ('source_node', 'destination_node')


@admin.register(NeuronMetrics)
class NeuronMetricsAdmin(admin.ModelAdmin):
    list_display = (
        'timestamp',
        'neuron_id',
        'energy_level',
        'congestion_level',
    )
    list_filter = ('timestamp', 'neuron_id')
    search_fields = ('neuron_id',)


@admin.register(NetworkGraphSnapshot)
class NetworkGraphSnapshotAdmin(admin.ModelAdmin):
    list_display = (
        'timestamp',
        'graph_data',
    )
    list_filter = ('timestamp',)
    search_fields = ('graph_data',)


@admin.register(PathHistory)
class PathHistoryAdmin(admin.ModelAdmin):
    list_display = (
        'timestamp',
        'path_type',
        'source_node',
        'destination_node',
        'path_nodes',
        'routing_path',
    )
    list_filter = ('timestamp', 'path_type', 'source_node', 'destination_node')
    search_fields = ('source_node', 'destination_node', 'path_type')