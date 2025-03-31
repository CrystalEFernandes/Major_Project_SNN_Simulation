from django.db import models

LOG_LEVEL_CHOICES = [
    ('DEBUG', 'Debug'),
    ('INFO', 'Info'),
    ('WARNING', 'Warning'),
    ('ERROR', 'Error'),
    ('CRITICAL', 'Critical'),
]

PATH_TYPE_CHOICES = [
    ('DIJKSTRA', 'Dijkstra'),
    ('STDP', 'STDP-guided'),
]

class SNNSettings(models.Model):
    learning_rate = models.FloatField(default=0.01)
    synaptic_weight = models.FloatField(default=0.005)
    firing_threshold = models.FloatField(default=-54)  # mV
    reset_potential = models.FloatField(default=-65)  # mV
    tau_pre = models.FloatField(default=20)  # ms
    tau_post = models.FloatField(default=20)  # ms
    simulation_time = models.FloatField(default=100)  # ms
    created_at = models.DateTimeField(auto_now_add=True)

class NetworkStatus(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    total_nodes = models.IntegerField()
    active_nodes = models.IntegerField()
    total_packets = models.IntegerField()
    avg_latency = models.FloatField()
    avg_energy = models.FloatField()
    avg_congestion = models.FloatField(default=0.0)

    def __str__(self):
        return f"Network Status - {self.timestamp}"

class RoutingPath(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    stdp_path = models.TextField()     # Or JSONField
    stdp_path_score = models.FloatField(null=True, blank=True) # Optional score
    source_node = models.IntegerField()
    destination_node = models.IntegerField()
    neighbor_scores_history_json = models.TextField(blank=True, null=True) 

    def __str__(self):
        return f"Routing Path from {self.source_node} to {self.destination_node} - {self.timestamp}"

class NeuronMetrics(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    neuron_id = models.IntegerField()
    energy_level = models.FloatField()
    congestion_level = models.FloatField()

    def __str__(self):
        return f"Neuron {self.neuron_id} Metrics - {self.timestamp}"

class NetworkGraphSnapshot(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    graph_data = models.JSONField()

    def __str__(self):
        return f"Network Graph Snapshot - {self.timestamp}"

class SimulationDataPoint(models.Model):
    timestamp = models.DateTimeField()
    simulation_time_ms = models.FloatField()
    firing_rates_json = models.TextField()
    latencies_ms_json = models.TextField()
    energy_levels_json = models.TextField()
    congestion_levels_json = models.TextField()
    routing_path = models.ForeignKey('RoutingPath', on_delete=models.SET_NULL, null=True, blank=True, related_name='simulation_data_points') # ForeignKey to RoutingPath

    def __str__(self):
        return f"Simulation Data Point - {self.timestamp}"

    class Meta:
        verbose_name = "Simulation Data Point"
        verbose_name_plural = "Simulation Data Points"

class PathHistory(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    path_type = models.CharField(max_length=10, choices=PATH_TYPE_CHOICES)
    path_nodes = models.JSONField() # Or TextField
    source_node = models.IntegerField()
    destination_node = models.IntegerField()
    routing_path = models.ForeignKey(RoutingPath, on_delete=models.SET_NULL, null=True, blank=True, related_name='path_history_entries') # Optional link

    def __str__(self):
        return f"Path History ({self.path_type}) - {self.timestamp} - {self.path_nodes}"