from django.shortcuts import render
import json
from typing import Dict, List, Set, Tuple
from .compare import OSPFSimulator, BGPSimulator, SNNRouter, TestScenario, TopologyType, SensorNode, NodeHardwareSpec, NodeSecuritySpec, EnvironmentalFactors
import time
import random
import numpy as np
from django.http import JsonResponse

# Create your views here.
def index(request):
    return render(request,'index.html')

def format_metric_name(metric):
    # Remove the unit suffixes
    name = metric.replace('_ms', '')
    name = name.replace('_mj', '')
    name = name.replace('_mb', '')
    name = name.replace('_kbps', '')
    
    # Replace underscores with spaces and capitalize each word
    words = name.split('_')
    return ' '.join(word.capitalize() for word in words)


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


def run_comprehensive_tests(scenarios):
    """Run comprehensive tests with different scenarios"""
    # scenarios = [
    #     (TopologyType.MESH, 10),
    #     # (TopologyType.STAR, 10),
    #     # (TopologyType.TREE, 10),
    #     # (TopologyType.RANDOM, 10),
    #     # (TopologyType.MESH, 20),
    #     # (TopologyType.STAR, 20),
    #     # (TopologyType.TREE, 20),
    #     # (TopologyType.RANDOM, 20)
    # ]
    
    results = {}
    for topology_type, size in scenarios:
        print(f"\nRunning tests for {topology_type.value} topology with {size} nodes...")
        scenario = TestScenario(topology_type, size)
        results[f"{topology_type.value}_{size}"] = scenario.run_comparison()
        
    return results

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




def routing_comparison_view(request):


    if request.method == 'POST':
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

            topology_type = request.POST.get('topology_type')
            node_count = int(request.POST.get('node_count', 0))
            
            # Validate inputs
            if not topology_type or node_count < 2 or node_count > 100:
                return JsonResponse({
                    'error': 'Invalid input. Please check your values.'
                }, status=400)
            
            # Create scenario tuple
            scenarios = [(TopologyType[topology_type], node_count)]
            
            # Run the tests
            results = run_comprehensive_tests(scenarios)
            # return JsonResponse({'results': results})
            # Convert the data structure to be more template-friendly
            print("\n=== Comprehensive Test Results ===")
            print("*" * 50)
            print(results)
            metrics_only = results[topology_type.lower() + "_" + str(node_count)]['metrics']
            

            # Convert results to JSON
            routing_data = convert_np_types(metrics_only)

            print(routing_data)
            metrics = list(routing_data["SNN"].keys())
            protocols = list(routing_data.keys())
            
            # Create a list of rows for the template
            rows = []
            for metric in metrics:
                row = {
                    'metric': format_metric_name(metric),  # Format the metric name
                    'original_metric': metric,  # Keep original metric name for conditional checks
                    'values': []
                }
                for protocol in protocols:
                    value = routing_data[protocol][metric]
                    row['values'].append({
                        'value': value,
                        'unit': 'ms' if 'time' in metric else 
                                'MB' if 'memory' in metric else 
                                'mJ' if 'energy' in metric else 
                                'Kbps' if 'throughput' in metric else '',
                        'decimals': 2 if 'time' in metric or 'ratio' in metric or 'energy' in metric or 'throughput' in metric else
                                3 if 'memory' in metric else 0
                    })
                rows.append(row)
            
            return render(request, 'routing_comparison.html', {
                'protocols': protocols,
                'rows': rows
            })
    else:
        return render(request, 'routing_comparison.html', {
            'protocols': [],
            'rows': []
        })















#routing_data = {
    #     "SNN": {
    #         "execution_time_ms": 2128.8132667541504,
    #         "path_length": 545,
    #         "energy_consumption_mj": 0.0,
    #         "convergence_time_ms": 0.0,
    #         "memory_usage_mb": 0.12109375,
    #         "route_stability": 0.0,
    #         "packet_delivery_ratio": 1.0,
    #         "control_overhead_packets": 89,
    #         "end_to_end_delay_ms": 0.0,
    #         "throughput_kbps": 0.0
    #     },
    #     "OSPF": {
    #         "execution_time_ms": 8.18014144897461,
    #         "path_length": 178,
    #         "energy_consumption_mj": 0.0,
    #         "convergence_time_ms": 0.0,
    #         "memory_usage_mb": 0.00390625,
    #         "route_stability": 0.0,
    #         "packet_delivery_ratio": 1.0,
    #         "control_overhead_packets": 89,
    #         "end_to_end_delay_ms": 0.0,
    #         "throughput_kbps": 0.0
    #     },
    #     "BGP": {
    #         "execution_time_ms": 60534.937381744385,
    #         "path_length": 178,
    #         "energy_consumption_mj": 0.0,
    #         "convergence_time_ms": 0.0,
    #         "memory_usage_mb": 9.37890625,
    #         "route_stability": 0.0,
    #         "packet_delivery_ratio": 1.0,
    #         "control_overhead_packets": 89,
    #         "end_to_end_delay_ms": 0.0,
    #         "throughput_kbps": 0.0
    #     }
    # }
    