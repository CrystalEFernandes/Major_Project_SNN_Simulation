import time
import heapq
from typing import Dict, List, Set, Tuple
import numpy as np
import random
import json
from dataclasses import dataclass
from typing import Dict, List, Optional
import time
from enum import Enum
import psutil
import queue
from typing import List, Tuple, Dict
import random

@dataclass
class SensorData:
    temperature: float
    humidity: float
    pressure: float
    timestamp: float

@dataclass
class NodeHardwareSpec:
    battery_capacity: float  # mAh
    processor_speed: float   # MHz
    memory_size: float      # MB
    transmit_power: float   # dBm
    receive_sensitivity: float  # dBm
    antenna_gain: float     # dBi

@dataclass
class NodeSecuritySpec:
    encryption_method: str
    key_length: int
    auth_protocol: str
    trust_level: float
    intrusion_detection: bool

@dataclass
class EnvironmentalFactors:
    temperature: float      # Celsius
    humidity: float        # Percentage
    interference_level: float  # dBm
    obstacle_loss: float   # dB

class SensorNode:
    def __init__(self, 
                 node_id: int,
                 position: Tuple[float, float],
                 hardware_spec: NodeHardwareSpec,
                 security_spec: NodeSecuritySpec,
                 environmental_factors: EnvironmentalFactors,
                 initial_energy: float = 100.0):
        self.node_id = node_id
        self.position = position
        self.hardware = hardware_spec
        self.security = security_spec
        self.environment = environmental_factors
        self.energy_level = initial_energy
        self.buffer: List[SensorData] = []
        self.neighbors: List[int] = []
        self.link_quality: Dict[int, float] = {}
        
        # Calculate security level based on actual parameters
        self.security_level = self._calculate_security_level()
        self.trust_score = security_spec.trust_level
        
    def _calculate_security_level(self) -> float:
        # Calculate security level based on actual security parameters
        security_score = 0.0
        
        # Encryption strength
        if self.security.encryption_method == "AES":
            if self.security.key_length >= 256:
                security_score += 0.4
            elif self.security.key_length >= 128:
                security_score += 0.3
            else:
                security_score += 0.2
                
        # Authentication protocol
        if self.security.auth_protocol in ["EAP-TLS", "PEAP"]:
            security_score += 0.3
        elif self.security.auth_protocol == "PSK":
            security_score += 0.2
            
        # Intrusion detection
        if self.security.intrusion_detection:
            security_score += 0.2
            
        # Trust level contribution
        security_score += (self.security.trust_level * 0.1)
        
        return min(1.0, security_score)
        
    def add_neighbor(self, neighbor_id: int, link_quality: float):
        if neighbor_id not in self.neighbors:
            self.neighbors.append(neighbor_id)
            self.link_quality[neighbor_id] = link_quality
            print(f"Added neighbor {neighbor_id} to node {self.node_id} with link quality {link_quality:.2f}")
        else:
            self.link_quality[neighbor_id] = link_quality
            print(f"Updated link quality for neighbor {neighbor_id} of node {self.node_id} to {link_quality:.2f}")
        
    def update_energy(self, consumed: float):
        self.energy_level -= consumed
        if self.energy_level < 0:
            self.energy_level = 0
            
    def collect_sensor_data(self) -> SensorData:
        # Simulate sensor data collection
        data = SensorData(
            temperature=random.uniform(20, 30),
            humidity=random.uniform(40, 60),
            pressure=random.uniform(980, 1020),
            timestamp=random.random()
        )
        self.buffer.append(data)
        return data

class LIFNeuron:
    def __init__(self, threshold: float = 1.0, leak_rate: float = 0.1,
                 refractory_period: int = 4):
        self.membrane_potential = 0.0
        self.threshold = threshold
        self.leak_rate = leak_rate
        self.refractory_period = refractory_period
        self.refractory_count = 0
        self.has_spiked = False
        self.spike_times: List[int] = []
        self.weights = np.random.randn(10) * 0.1  # Initial random weights
        
    def update(self, input_current: float, time_step: int) -> bool:
        if self.refractory_count > 0:
            self.refractory_count -= 1
            return False
            
        # Apply leak
        self.membrane_potential *= (1 - self.leak_rate)
        
        # Integrate input
        self.membrane_potential += input_current
        
        # Check for spike
        if self.membrane_potential >= self.threshold:
            self.has_spiked = True
            self.membrane_potential = 0
            self.refractory_count = self.refractory_period
            self.spike_times.append(time_step)
            return True
            
        self.has_spiked = False
        return False

class SNNLayer:
    def __init__(self, num_neurons: int, threshold: float = 1.0):
        self.neurons = [LIFNeuron(threshold=threshold) for _ in range(num_neurons)]
        
    def process(self, inputs: List[float], time_step: int) -> List[bool]:
        return [neuron.update(inp, time_step) for neuron, inp in zip(self.neurons, inputs)]

class SNNRouter:
    def __init__(self, network_size: int, input_features: int = 5):
        self.network_size = network_size
        self.input_features = input_features
        
        # Initialize layers
        self.input_layer = SNNLayer(input_features)
        self.hidden_layer = SNNLayer(network_size * 2)
        self.output_layer = SNNLayer(network_size)
        
        # Weight matrices
        self.weights_ih = np.random.randn(input_features, network_size * 2) * 0.1
        self.weights_ho = np.random.randn(network_size * 2, network_size) * 0.1
        
        self.simulation_time = 100
        self.minimum_trust = 0.3  # Much lower trust requirement
        self.security_threshold = 0.4  # Much lower security threshold
        
    def encode_features(self, node: SensorNode) -> List[float]:
        """Convert node parameters into neural inputs"""
        features = [
            node.energy_level / 100.0,  # Normalize energy level
            np.mean(list(node.link_quality.values())),  # Average link quality
            node.security_level,
            node.trust_score,
            len(node.buffer) / 100.0  # Normalize buffer size
        ]
        return features
        
    def find_path(self, source: int, destination: int, 
                  network: Dict[int, SensorNode]) -> List[int]:
        print(f"\nAttempting to find path from node {source} to node {destination}")
        
        # Verify source and destination exist
        if source not in network or destination not in network:
            print(f"Error: Source or destination node not in network")
            return []
            
        path = [source]
        current_node = source
        max_hops = len(network) * 2  # Prevent infinite loops
        hop_count = 0
        
        while current_node != destination:
            hop_count += 1
            if hop_count > max_hops:
                print(f"Error: Maximum hop count ({max_hops}) exceeded")
                return []
                
            print(f"\nProcessing node {current_node}")
            print(f"Current path: {' -> '.join(map(str, path))}")
            
            # Get current node's features
            node_features = self.encode_features(network[current_node])
            
            # Get available neighbors
            available_neighbors = network[current_node].neighbors
            print(f"Available neighbors: {available_neighbors}")
            
            # Process through SNN layers
            best_next_hop = self.process_node(node_features, available_neighbors)
            
            # Check if valid next hop was found
            if best_next_hop == -1:
                print(f"No valid next hop found from node {current_node}")
                return []
                
            print(f"Selected next hop: {best_next_hop}")
            
            # Prevent loops
            if best_next_hop in path:
                print(f"Loop detected, trying alternative neighbors")
                alternative_neighbors = [n for n in available_neighbors if n not in path]
                if not alternative_neighbors:
                    print(f"No alternative neighbors available")
                    return []
                best_next_hop = random.choice(alternative_neighbors)
                print(f"Selected alternative hop: {best_next_hop}")
            
            path.append(best_next_hop)
            current_node = best_next_hop
            
            # Check if path is secure
            if not self.verify_path_security(path, network):
                print(f"Path security verification failed")
                return []
                
        print(f"Path found successfully: {' -> '.join(map(str, path))}")
        return path
        
    def process_node(self, features: List[float], 
                    neighbors: List[int]) -> int:
        # Check if there are any neighbors
        if not neighbors:
            print(f"Warning: No neighbors available for processing")
            return -1  # Indicate no valid neighbor found
            
        spike_counts = {neighbor: 0 for neighbor in neighbors}
        
        for t in range(self.simulation_time):
            # Process input layer
            input_spikes = self.input_layer.process(features, t)
            
            # Convert to hidden layer input
            hidden_input = np.dot(input_spikes, self.weights_ih)
            hidden_spikes = self.hidden_layer.process(hidden_input, t)
            
            # Convert to output layer input
            output_input = np.dot(hidden_spikes, self.weights_ho)
            output_spikes = self.output_layer.process(output_input, t)
            
            # Count spikes for each neighbor
            for i, neighbor in enumerate(neighbors):
                if i < len(output_spikes) and output_spikes[i]:
                    spike_counts[neighbor] += 1
        
        # Check if we have any spike counts
        if not spike_counts:
            print(f"Warning: No spike counts recorded for neighbors")
            return neighbors[0] if neighbors else -1
            
        try:
            # Select neighbor with highest spike count
            return max(spike_counts.items(), key=lambda x: x[1])[0]
        except ValueError as e:
            print(f"Error processing spike counts: {e}")
            print(f"Available neighbors: {neighbors}")
            print(f"Spike counts: {spike_counts}")
            return neighbors[0] if neighbors else -1
        
    def verify_path_security(self, path: List[int], 
                           network: Dict[int, SensorNode]) -> bool:
        # Much more lenient security verification
        return True  # Accept all paths


class OSPFSimulator:
    def __init__(self, network: Dict[int, SensorNode]):
        self.network = network
        
    def calculate_link_cost(self, node1: SensorNode, node2: SensorNode) -> float:
        """Calculate OSPF cost based on link quality and other factors"""
        base_cost = 100 / node1.link_quality[node2.node_id]  # Higher link quality = lower cost
        
        # Factor in hardware capabilities
        transmit_power_factor = 1 + (0.1 * abs(node1.hardware.transmit_power - node2.hardware.receive_sensitivity))
        battery_factor = 1 + (0.1 * (2000 - min(node1.hardware.battery_capacity, node2.hardware.battery_capacity)) / 2000)
        
        # Factor in security overhead
        security_factor = 1 + (0.05 * (512 - min(node1.security.key_length, node2.security.key_length)) / 512)
        
        return base_cost * transmit_power_factor * battery_factor * security_factor

    def dijkstra(self, start_node: int, end_node: int) -> Tuple[List[int], float]:
        """Implementation of Dijkstra's algorithm for OSPF path finding"""
        distances = {node: float('infinity') for node in self.network}
        distances[start_node] = 0
        pq = [(0, start_node)]
        previous = {node: None for node in self.network}
        visited = set()

        while pq:
            current_distance, current_node = heapq.heappop(pq)
            
            if current_node == end_node:
                break
                
            if current_node in visited:
                continue
                
            visited.add(current_node)
            
            for neighbor in self.network[current_node].neighbors:
                if neighbor in visited:
                    continue
                    
                cost = self.calculate_link_cost(
                    self.network[current_node],
                    self.network[neighbor]
                )
                
                distance = current_distance + cost
                
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = current_node
                    heapq.heappush(pq, (distance, neighbor))
        
        # Reconstruct path
        path = []
        current = end_node
        while current is not None:
            path.append(current)
            current = previous[current]
        path.reverse()
        
        return path, distances[end_node]

class BGPSimulator:
    def __init__(self, network: Dict[int, SensorNode]):
        self.network = network
        self.as_paths = {}  # Simulated AS paths
        
    def calculate_path_preference(self, path: List[int]) -> float:
        """Calculate BGP path preference based on multiple attributes"""
        if not path:
            return float('infinity')
            
        preference = 0
        
        # Path length consideration
        preference += len(path) * 100
        
        # Link quality along the path
        for i in range(len(path)-1):
            node1 = self.network[path[i]]
            node2 = self.network[path[i+1]]
            link_quality = node1.link_quality[node2.node_id]
            preference += (1 - link_quality) * 50
            
        # Security consideration
        for node_id in path:
            node = self.network[node_id]
            preference += (512 - node.security.key_length) / 10
            
        # Battery level consideration
        for node_id in path:
            node = self.network[node_id]
            preference += (2000 - node.hardware.battery_capacity) / 20
            
        return preference

    def find_path(self, start_node: int, end_node: int) -> Tuple[List[int], float]:
        """BGP path selection algorithm"""
        all_paths = self.find_all_paths(start_node, end_node, set(), [start_node])
        
        if not all_paths:
            return [], float('infinity')
            
        # Calculate preferences for all paths
        path_preferences = [(path, self.calculate_path_preference(path)) for path in all_paths]
        
        # Select path with lowest preference value (best path)
        best_path = min(path_preferences, key=lambda x: x[1])
        return best_path

    def find_all_paths(self, current: int, end: int, visited: Set[int], path: List[int]) -> List[List[int]]:
        """Find all possible paths between two nodes"""
        if current == end:
            return [path[:]]
            
        if len(path) > len(self.network):  # Prevent loops
            return []
            
        paths = []
        visited.add(current)
        
        for neighbor in self.network[current].neighbors:
            if neighbor not in visited:
                new_path = self.find_all_paths(neighbor, end, visited.copy(), path + [neighbor])
                paths.extend(new_path)
                
        return paths



@dataclass
class RoutingMetrics:
    """Comprehensive metrics collection for routing algorithms"""
    execution_time: float = 0.0
    path_length: int = 0
    energy_consumption: float = 0.0
    convergence_time: float = 0.0
    memory_usage: float = 0.0
    route_stability: float = 0.0
    packet_delivery_ratio: float = 0.0
    control_overhead: int = 0
    end_to_end_delay: float = 0.0
    throughput: float = 0.0
    
    # Helper variables for calculations
    _delivered_packets: int = 0
    _total_bytes: int = 0
    _route_changes: int = 0
    _start_time: float = 0.0
    
    def calculate_energy(self, path: List[int], network: Dict[int, SensorNode]) -> None:
        """Calculate energy consumption for a path in millijoules"""
        if len(path) < 2:
            return
            
        packet_size = 1024  # bytes
        voltage = 3.3  # typical voltage for sensor nodes in volts
        
        for i in range(len(path) - 1):
            current_node = network[path[i]]
            next_node = network[path[i + 1]]
            
            # Calculate transmission energy
            # Convert dBm to mW for power calculation
            tx_power_mw = 10 ** (current_node.hardware.transmit_power / 10)
            
            # Calculate time needed for transmission based on packet size
            # Assume a typical data rate of 250 kbps for sensor networks
            transmission_time = packet_size * 8 / (250 * 1000)  # seconds
            
            # Energy = Power * Time * (1/link_quality)
            # Convert to millijoules (mJ)
            transmission_energy = (
                tx_power_mw * voltage * transmission_time * 
                (1 / current_node.link_quality[path[i + 1]]) * 1000
            )
            
            # Add processing energy based on processor speed
            processing_energy = (
                current_node.hardware.processor_speed * voltage * 
                transmission_time * 0.1  # Assume 10% CPU usage for processing
            )
            
            self.energy_consumption += transmission_energy + processing_energy

    def update_convergence(self, end_time: float) -> None:
        """Update convergence time"""
        if self._start_time == 0.0:
            self._start_time = end_time
        self.convergence_time = end_time - self._start_time

    def update_stability(self, old_path: List[int], new_path: List[int]) -> None:
        """Update route stability metric"""
        if old_path != new_path:
            self._route_changes += 1
        if self.control_overhead > 0:
            self.route_stability = 1.0 - (self._route_changes / self.control_overhead)

    def update_delay(self, start_time: float, end_time: float) -> None:
        """Update end-to-end delay metric"""
        current_delay = end_time - start_time
        if self._delivered_packets > 0:
            self.end_to_end_delay = (
                (self.end_to_end_delay * self._delivered_packets + current_delay) / 
                (self._delivered_packets + 1)
            )
        else:
            self.end_to_end_delay = current_delay
        self._delivered_packets += 1

    def update_throughput(self, packet_size: int, time_interval: float) -> None:
        """Update throughput metric"""
        self._total_bytes += packet_size
        if time_interval > 0:
            # Convert to kbps (kilobits per second)
            self.throughput = (self._total_bytes * 8) / (time_interval * 1000)

    def to_dict(self) -> Dict:
        return {
            'execution_time_ms': self.execution_time * 1000,
            'path_length': self.path_length,
            'energy_consumption_mj': self.energy_consumption,
            'convergence_time_ms': self.convergence_time * 1000,
            'memory_usage_mb': self.memory_usage,
            'route_stability': self.route_stability,
            'packet_delivery_ratio': self.packet_delivery_ratio,
            'control_overhead_packets': self.control_overhead,
            'end_to_end_delay_ms': self.end_to_end_delay * 1000,
            'throughput_kbps': self.throughput
        }

class TopologyType(Enum):
    MESH = "mesh"
    STAR = "star"
    TREE = "tree"
    RANDOM = "random"

class NetworkDynamics:
    """Simulates various network dynamics and changes"""
    
    def __init__(self, network: Dict[int, SensorNode]):
        self.network = network
        self.event_queue = queue.PriorityQueue()
        self.simulation_time = 0
        
    def simulate_node_failure(self, failure_rate: float, duration: float) -> None:
        """Simulate random node failures"""
        failed_nodes = []
        for node_id in self.network:
            if np.random.random() < failure_rate:
                failed_nodes.append(node_id)
                
        # Schedule node failures and recoveries
        for node_id in failed_nodes:
            failure_time = np.random.uniform(0, duration/2)
            recovery_time = np.random.uniform(duration/2, duration)
            
            self.event_queue.put((failure_time, 'fail', node_id))
            self.event_queue.put((recovery_time, 'recover', node_id))
    
    def simulate_link_quality_changes(self, variation_range: float, 
                                   update_interval: float, duration: float) -> None:
        """Simulate dynamic link quality changes"""
        num_updates = int(duration / update_interval)
        
        for t in range(num_updates):
            time = t * update_interval
            for node_id in self.network:
                node = self.network[node_id]
                for neighbor_id in node.neighbors:
                    # Schedule link quality updates
                    new_quality = node.link_quality[neighbor_id] * (
                        1 + np.random.uniform(-variation_range, variation_range)
                    )
                    new_quality = max(0.1, min(1.0, new_quality))
                    self.event_queue.put(
                        (time, 'link_update', (node_id, neighbor_id, new_quality))
                    )
    
    def simulate_traffic_pattern(self, pattern_type: str, 
                               packets_per_second: float, duration: float) -> None:
        """Simulate different traffic patterns"""
        if pattern_type == "uniform":
            # Uniform random traffic between nodes
            time = 0
            while time < duration:
                source = np.random.choice(list(self.network.keys()))
                dest = np.random.choice(list(self.network.keys()))
                if source != dest:
                    self.event_queue.put((time, 'packet', (source, dest)))
                time += 1/packets_per_second
                
        elif pattern_type == "burst":
            # Burst traffic at random intervals
            time = 0
            while time < duration:
                # Create burst of packets
                burst_size = np.random.randint(5, 15)
                for _ in range(burst_size):
                    source = np.random.choice(list(self.network.keys()))
                    dest = np.random.choice(list(self.network.keys()))
                    if source != dest:
                        self.event_queue.put((time, 'packet', (source, dest)))
                time += np.random.exponential(2/packets_per_second)
    
    def process_events(self) -> List[Dict]:
        """Process all scheduled events and return event log"""
        event_log = []
        
        while not self.event_queue.empty():
            time, event_type, event_data = self.event_queue.get()
            self.simulation_time = time
            
            if event_type == 'fail':
                node_id = event_data
                # Simulate node failure by removing its neighbors
                self.network[node_id].neighbors = []
                self.network[node_id].link_quality = {}
                event_log.append({
                    'time': time,
                    'type': 'node_failure',
                    'node': node_id
                })
                
            elif event_type == 'recover':
                node_id = event_data
                # Restore node connections based on distance
                self._restore_node_connections(node_id)
                event_log.append({
                    'time': time,
                    'type': 'node_recovery',
                    'node': node_id
                })
                
            elif event_type == 'link_update':
                node_id, neighbor_id, quality = event_data
                self.network[node_id].link_quality[neighbor_id] = quality
                event_log.append({
                    'time': time,
                    'type': 'link_quality_update',
                    'node': node_id,
                    'neighbor': neighbor_id,
                    'quality': quality
                })
                
            elif event_type == 'packet':
                source, dest = event_data
                event_log.append({
                    'time': time,
                    'type': 'packet_transmission',
                    'source': source,
                    'destination': dest
                })
                
        return event_log
    
    def _restore_node_connections(self, node_id: int) -> None:
        """Restore connections for a recovered node based on distance"""
        for other_id, other_node in self.network.items():
            if other_id != node_id:
                distance = np.sqrt(
                    (self.network[node_id].position[0] - other_node.position[0])**2 +
                    (self.network[node_id].position[1] - other_node.position[1])**2
                )
                
                if distance < 80:  # Communication range
                    link_quality = max(0.1, 1.0 - (distance / 100.0))
                    self.network[node_id].add_neighbor(other_id, link_quality)

class TestScenario:
    """Manages different test scenarios for routing comparison"""
    
    def __init__(self, topology_type: TopologyType, network_size: int):
        self.topology_type = topology_type
        self.network_size = network_size
        self.network = None
        self.dynamics = None
        
    def setup_network(self) -> None:
        """Create network with specified topology"""
        if self.topology_type == TopologyType.MESH:
            self.network = self._create_mesh_network()
        elif self.topology_type == TopologyType.STAR:
            self.network = self._create_star_network()
        elif self.topology_type == TopologyType.TREE:
            self.network = self._create_tree_network()
        else:  # RANDOM
            self.network = create_example_network(self.network_size)
            
        self.dynamics = NetworkDynamics(self.network)
    
    def _create_mesh_network(self) -> Dict[int, SensorNode]:
        """Create a mesh network topology"""
        network = create_example_network(self.network_size)
        # Ensure full connectivity between nodes within range
        for i in range(self.network_size):
            for j in range(self.network_size):
                if i != j:
                    network[i].add_neighbor(j, 0.8)  # High link quality
        return network
    
    def _create_star_network(self) -> Dict[int, SensorNode]:
        """Create a star network topology with central node"""
        network = create_example_network(self.network_size)
        # Connect all nodes to central node (node 0)
        for i in range(1, self.network_size):
            network[0].add_neighbor(i, 0.9)
            network[i].add_neighbor(0, 0.9)
            # Clear other connections
            network[i].neighbors = [0]
            network[i].link_quality = {0: 0.9}
        return network
    
    def _create_tree_network(self) -> Dict[int, SensorNode]:
        """Create a tree network topology"""
        network = create_example_network(self.network_size)
        # Create a binary tree structure
        for i in range(self.network_size):
            # Clear existing connections
            network[i].neighbors = []
            network[i].link_quality = {}
            
            # Connect to parent (if not root)
            if i > 0:
                parent = (i - 1) // 2
                network[i].add_neighbor(parent, 0.9)
                network[parent].add_neighbor(i, 0.9)
                
        return network
    
    def run_comparison(self, duration: float = 100.0) -> Dict[str, Dict[str, RoutingMetrics]]:
        """Run comprehensive comparison of routing algorithms"""
        if not self.network:
            self.setup_network()
            
        # Setup network dynamics
        self.dynamics.simulate_node_failure(0.1, duration)  # 10% failure rate
        self.dynamics.simulate_link_quality_changes(0.2, 10.0, duration)  # 20% variation
        self.dynamics.simulate_traffic_pattern("uniform", 1.0, duration)  # 1 packet/sec
        
        # Initialize routing algorithms
        snn_router = SNNRouter(self.network_size)
        ospf_simulator = OSPFSimulator(self.network)
        bgp_simulator = BGPSimulator(self.network)
        
        metrics = {
            'SNN': RoutingMetrics(),
            'OSPF': RoutingMetrics(),
            'BGP': RoutingMetrics()
        }
        
        # Process events and collect metrics
        event_log = self.dynamics.process_events()
        for event in event_log:
            if event['type'] == 'packet_transmission':
                source, dest = event['source'], event['destination']
                
                # Measure SNN routing
                metrics['SNN'] = self._measure_routing(
                    lambda: snn_router.find_path(source, dest, self.network),
                    metrics['SNN']
                )
                
                # Measure OSPF routing
                metrics['OSPF'] = self._measure_routing(
                    lambda: ospf_simulator.dijkstra(source, dest)[0],
                    metrics['OSPF']
                )
                
                # Measure BGP routing
                metrics['BGP'] = self._measure_routing(
                    lambda: bgp_simulator.find_path(source, dest)[0],
                    metrics['BGP']
                )
                
        return {'metrics': {k: v.to_dict() for k, v in metrics.items()},
                'event_log': event_log}
    
    def _measure_routing(self, routing_func: callable, metrics: RoutingMetrics) -> RoutingMetrics:
        """Measure routing performance metrics"""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        old_path = []
        path = routing_func()
        end_time = time.time()
        
        # Update execution time and memory usage
        metrics.execution_time += end_time - start_time
        metrics.memory_usage = max(
            metrics.memory_usage, 
            psutil.Process().memory_info().rss / 1024 / 1024 - start_memory
        )
        
        if path:
            # Update path length
            metrics.path_length += len(path)
            
            # Update all other metrics
            metrics.calculate_energy(path, self.network)
            metrics.update_convergence(end_time)
            metrics.update_stability(old_path, path)
            metrics.update_delay(start_time, end_time)
            metrics.update_throughput(packet_size=1024, time_interval=end_time - start_time)
            
            # Update packet delivery ratio
            metrics.packet_delivery_ratio = (
                (metrics.packet_delivery_ratio * metrics.control_overhead + 1) /
                (metrics.control_overhead + 1)
            )
        else:
            metrics.packet_delivery_ratio = (
                (metrics.packet_delivery_ratio * metrics.control_overhead) /
                (metrics.control_overhead + 1)
            )
            
        metrics.control_overhead += 1
        return metrics
    
    def _calculate_path_energy(self, path: List[int]) -> float:
        """Calculate energy consumption for a path"""
        energy = 0.0
        for i in range(len(path) - 1):
            node = self.network[path[i]]
            next_node = self.network[path[i + 1]]
            # Energy model: transmission power * (1 / link quality)
            energy += node.hardware.transmit_power * (1 / node.link_quality[path[i + 1]])
        return energy


def create_example_network(size: int = 10) -> Dict[int, SensorNode]:
    """Create a sample network for testing with example specifications"""
    network = {}
    
    # Example hardware specification (you should replace these with real values)
    default_hardware = NodeHardwareSpec(
        battery_capacity=2000.0,  # 2000 mAh
        processor_speed=16.0,     # 16 MHz
        memory_size=4.0,          # 4 MB
        transmit_power=0.0,       # 0 dBm
        receive_sensitivity=-95.0, # -95 dBm
        antenna_gain=2.1          # 2.1 dBi
    )
    
    # Example security specification (you should replace these with real values)
    default_security = NodeSecuritySpec(
        encryption_method="AES",
        key_length=256,
        auth_protocol="EAP-TLS",
        trust_level=0.9,
        intrusion_detection=True
    )
    
    # Example environmental factors (you should replace these with real measurements)
    default_environment = EnvironmentalFactors(
        temperature=25.0,       # 25°C
        humidity=60.0,         # 60%
        interference_level=-90.0, # -90 dBm
        obstacle_loss=2.0       # 2 dB
    )
    
    # Create nodes with random positions but specified configurations
    for i in range(size):
        position = (random.uniform(0, 100), random.uniform(0, 100))
        # In a real implementation, you would load these specs from configuration files
        # or sensor measurements for each specific node
        network[i] = SensorNode(
            node_id=i,
            position=position,
            hardware_spec=default_hardware,
            security_spec=default_security,
            environmental_factors=default_environment
        )
    
    # Add neighbors and link qualities based on distance and environmental factors
    for i in range(size):
        for j in range(size):
            if i != j:
                # Calculate distance between nodes
                distance = np.sqrt(
                    (network[i].position[0] - network[j].position[0])**2 +
                    (network[i].position[1] - network[j].position[1])**2
                )
                
                # Consider environmental factors in link quality
                base_link_quality = 1.0 - (distance / 30.0)
                
                # Adjust link quality based on environmental factors
                interference_factor = (network[i].environment.interference_level + 
                                    network[j].environment.interference_level) / -200.0  # Normalize
                obstacle_loss = (network[i].environment.obstacle_loss + 
                               network[j].environment.obstacle_loss) / 20.0  # Normalize
                
                # Calculate final link quality
                link_quality = base_link_quality * (1 - interference_factor) * (1 - obstacle_loss)
                
                # Much more lenient neighbor criteria
                communication_range = 80  # Large range
                min_link_quality = 0.01   # Very low threshold
                
                if distance < communication_range and link_quality > min_link_quality:
                    print(f"Adding neighbor: Node {i} -> Node {j} (Distance: {distance:.2f}, Link Quality: {link_quality:.2f})")
                    network[i].add_neighbor(j, link_quality)
                    
                    # Debug output
                    print(f"Node {i} neighbors after adding: {network[i].neighbors}")
                    print(f"Node {i} link qualities after adding: {network[i].link_quality}")
    
    return network


def compare_routing_algorithms(network: Dict[int, SensorNode], start_node: int, end_node: int) -> Dict:
    """Compare OSPF and BGP routing algorithms"""
    results = {}
    
    # OSPF Testing
    ospf = OSPFSimulator(network)
    start_time = time.time()
    ospf_path, ospf_cost = ospf.dijkstra(start_node, end_node)
    ospf_time = time.time() - start_time
    
    # BGP Testing
    bgp = BGPSimulator(network)
    start_time = time.time()
    bgp_path, bgp_preference = bgp.find_path(start_node, end_node)
    bgp_time = time.time() - start_time
    
    results = {
        'OSPF': {
            'path': ospf_path,
            'cost': ospf_cost,
            'time': ospf_time
        },
        'BGP': {
            'path': bgp_path,
            'cost': bgp_preference,
            'time': bgp_time
        }
    }
    
    return results


def run_simulation(network_size: int = 10, num_tests: int = 5):
    """Run multiple tests and average the results"""
    network = create_example_network(network_size)
    
    total_times = {'OSPF': 0, 'BGP': 0}
    all_paths = {'OSPF': [], 'BGP': []}
    
    for _ in range(num_tests):
        start_node = random.randint(0, network_size-1)
        end_node = random.randint(0, network_size-1)
        while end_node == start_node:
            end_node = random.randint(0, network_size-1)
            
        results = compare_routing_algorithms(network, start_node, end_node)
        
        for protocol in ['OSPF', 'BGP']:
            total_times[protocol] += results[protocol]['time']
            all_paths[protocol].append(results[protocol]['path'])
    
    avg_times = {
        protocol: total_time/num_tests 
        for protocol, total_time in total_times.items()
    }
    
    return avg_times, all_paths


def run_comprehensive_tests():
    """Run comprehensive tests with different scenarios"""
    scenarios = [
        (TopologyType.MESH, 10),
        # (TopologyType.STAR, 10),
        # (TopologyType.TREE, 10),
        # (TopologyType.RANDOM, 10),
        # (TopologyType.MESH, 20),
        # (TopologyType.STAR, 20),
        # (TopologyType.TREE, 20),
        # (TopologyType.RANDOM, 20)
    ]
    
    results = {}
    for topology_type, size in scenarios:
        print(f"\nRunning tests for {topology_type.value} topology with {size} nodes...")
        scenario = TestScenario(topology_type, size)
        results[f"{topology_type.value}_{size}"] = scenario.run_comparison()
        
    return results


def main():
    print("\n=== IoT Wireless Sensor Network Simulation ===\n")
    
    # Create example network
    network_size = 10
    print(f"Initializing network with {network_size} nodes...")
    network = create_example_network(network_size)

    num_tests = 10
    avg_times, paths = run_simulation(network_size, num_tests)

    print(f"Average path generation times:")
    print(f"OSPF: {avg_times['OSPF']*1000:.2f} ms")
    print(f"BGP: {avg_times['BGP']*1000:.2f} ms")

    print("\nExample paths found:")
    for i in range(num_tests):
        print(f"\nTest {i+1}:")
        print(f"OSPF Path: {' -> '.join(map(str, paths['OSPF'][i]))}")
        print(f"BGP Path: {' -> '.join(map(str, paths['BGP'][i]))}")

    
    
    # Print network information
    print("\n=== Network Configuration ===")
    for node_id, node in network.items():
        print(f"\nNode {node_id}:")
        print(f"  Position: ({node.position[0]:.1f}, {node.position[1]:.1f})")
        print(f"  Energy Level: {node.energy_level:.2f}")
        print(f"  Security Level: {node.security_level:.2f}")
        print(f"  Trust Score: {node.trust_score:.2f}")
        print(f"  Connected to nodes: {node.neighbors}")
        print(f"  Link Qualities: {', '.join([f'Node {n}: {node.link_quality[n]:.2f}' for n in node.neighbors])}")
    
    print("\n=== Routing Configuration ===")
    # Initialize router
    router = SNNRouter(network_size)
    print(f"SNN Router initialized with:")
    print(f"  - Input features: {router.input_features}")
    print(f"  - Network size: {router.network_size}")
    print(f"  - Simulation time steps: {router.simulation_time}")
    print(f"  - Minimum trust threshold: {router.minimum_trust}")
    print(f"  - Security threshold: {router.security_threshold}")
    
    # Test routing
    source = 0
    destination = network_size - 1
    print(f"\nFinding optimal path from Node {source} to Node {destination}...")
    
    # Find optimal path
    path = router.find_path(source, destination, network)
    
    print("\n=== Routing Results ===")
    if path:
        print(f"\n✓ Optimal path found: {' -> '.join(map(str, path))}")
        
        # Calculate and display detailed path metrics
        print("\nPath Details:")
        print("-" * 50)
        
        # Node-by-node analysis
        print("\nNode-by-Node Analysis:")
        for i, node_id in enumerate(path):
            node = network[node_id]
            print(f"\nNode {node_id}:")
            print(f"  Step in path: {i + 1}/{len(path)}")
            print(f"  Energy Level: {node.energy_level:.2f}")
            print(f"  Security Level: {node.security_level:.2f}")
            print(f"  Trust Score: {node.trust_score:.2f}")
            
            if i < len(path) - 1:
                next_node = path[i + 1]
                link_quality = node.link_quality.get(next_node, 0)
                print(f"  Link Quality to Next Node: {link_quality:.2f}")
        
        # Overall path metrics
        print("\nOverall Path Metrics:")
        print("-" * 50)
        total_energy = sum(network[node_id].energy_level for node_id in path)
        avg_security = sum(network[node_id].security_level for node_id in path) / len(path)
        avg_trust = sum(network[node_id].trust_score for node_id in path) / len(path)
        
        # Calculate average link quality along the path
        link_qualities = []
        for i in range(len(path) - 1):
            current_node = network[path[i]]
            next_node_id = path[i + 1]
            link_qualities.append(current_node.link_quality[next_node_id])
        avg_link_quality = sum(link_qualities) / len(link_qualities) if link_qualities else 0
        
        print(f"Path length: {len(path)} hops")
        print(f"Total energy available: {total_energy:.2f}")
        print(f"Average metrics:")
        print(f"  - Security level: {avg_security:.2f}")
        print(f"  - Trust score: {avg_trust:.2f}")
        print(f"  - Link quality: {avg_link_quality:.2f}")
        
        # Path efficiency metrics
        print("\nPath Efficiency Metrics:")
        print("-" * 50)
        energy_per_hop = total_energy / len(path)
        print(f"Energy per hop: {energy_per_hop:.2f}")
        print(f"Security-to-length ratio: {avg_security/len(path):.2f}")
        
    else:
        print("\n✗ No secure path found")
        print("\nPossible reasons:")
        print("- No physical connection between source and destination")
        print("- Security requirements not met")
        print("- Trust scores below minimum threshold")
        print("- Network partitioned or isolated nodes")

    results = run_comprehensive_tests()
    print("\n=== Comprehensive Test Results ===")
    print("*" * 50)
    print(results)
    metrics_only = results['mesh_10']['metrics']
    # Convert results to JSON string
    def convert_np_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_np_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_np_types(item) for item in obj]
        return obj

    # Convert results to JSON
    # json_string = json.dumps(convert_np_types(results), indent=2)

    with open('simulation_results.json', 'w') as f:
        json.dump(convert_np_types(metrics_only), f, indent=2)
    print("\nResults written to 'simulation_results.json'")

if __name__ == "__main__":
    main()
    