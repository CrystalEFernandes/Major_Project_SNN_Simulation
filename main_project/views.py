from django.shortcuts import render, get_object_or_404, redirect
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from .models import NetworkStatus, RoutingPath, NetworkGraphSnapshot, NeuronMetrics, PathHistory, PATH_TYPE_CHOICES, SNNSettings, SimulationDataPoint # Import SimulationDataPoint model
import json
from django.utils import timezone
import logging
import google.generativeai as genai
import os
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
load_dotenv()

api_key_value = os.environ.get("GOOGLE_API_KEY")
if api_key_value:
    print("GOOGLE_API_KEY environment variable is set and found in views.py!")
    print(f"First 5 characters of API key: {api_key_value[:5]}") # Print first 5 chars (for security - don't print full key)
else:
    print("GOOGLE_API_KEY environment variable NOT FOUND in views.py!")

logger = logging.getLogger(__name__)

REPORT_STORAGE_DIR = os.path.join(settings.BASE_DIR, 'reports')
os.makedirs(REPORT_STORAGE_DIR, exist_ok=True)

def get_report_file_path(simulation_id):
    """
    Helper function to construct the file path for a simulation report.
    """
    return os.path.join(REPORT_STORAGE_DIR, f"report_simulation_{simulation_id}.txt")

@csrf_exempt
def update_routing(request): # This view is likely not used anymore, as we moved to receive_simulation_data
    """
    Endpoint to receive routing updates from the SNN script (non-REST version).
    Saves RoutingPath, NetworkGraphSnapshot, NeuronMetrics, and PathHistory data.
    """
    if request.method == 'POST':
        try:
            data = json.loads(request.body.decode('utf-8')) # Parse JSON data from request body
            dijkstra_path_nodes = data.get('dijkstra_path', [])
            selected_path_nodes = data.get('selected_path', [])
            graph_data = data.get('graph', {})
            energy_levels = data.get('energy_levels', []) # These energy_levels and congestion_levels are neuron metrics, might be redundant here
            congestion_levels = data.get('congestion_levels', [])

            # Save Routing Path
            routing_path_instance = RoutingPath.objects.create(
                dijkstra_path=json.dumps(dijkstra_path_nodes),
                stdp_path=json.dumps(selected_path_nodes),
                source_node=0,  # Assuming source is always node 0 as per script logic
                destination_node=len(energy_levels) - 1 if energy_levels else 4 # Assuming destination is last node (N-1) or 4 if levels are empty
            )

            # Save Network Graph Snapshot
            NetworkGraphSnapshot.objects.create(graph_data=graph_data)

            # Save Neuron Metrics (Consider if this is redundant, as simulation_data endpoint also saves neuron metrics)
            for i in range(len(energy_levels)):
                NeuronMetrics.objects.create(
                    neuron_id=i,
                    energy_level=energy_levels[i],
                    congestion_level=congestion_levels[i]
                )

            # Save Path History for Dijkstra
            PathHistory.objects.create(
                path_type=PATH_TYPE_CHOICES[0][0], # 'DIJKSTRA'
                path_nodes=json.dumps(dijkstra_path_nodes),
                source_node=0,
                destination_node=len(energy_levels) - 1 if energy_levels else 4,
                routing_path=routing_path_instance
            )

            # Save Path History for STDP
            PathHistory.objects.create(
                path_type=PATH_TYPE_CHOICES[1][0], # 'STDP'
                path_nodes=json.dumps(selected_path_nodes),
                source_node=0,
                destination_node=len(energy_levels) - 1 if energy_levels else 4,
                routing_path=routing_path_instance
            )

            return JsonResponse({"message": "Routing data updated successfully."}, status=201) # Use JsonResponse for JSON response

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON format in request."}, status=400)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)
    else:
        return JsonResponse({"error": "Invalid request method."}, status=405)

@csrf_exempt # Exempt from CSRF protection for API endpoint
def update_status(request):
    """
    Endpoint to receive network status updates.
    Saves NetworkStatus data.
    """
    if request.method == 'POST':
        try:
            data = json.loads(request.body.decode('utf-8'))
            NetworkStatus.objects.create(
                total_nodes=data.get('total_nodes'),
                active_nodes=data.get('active_nodes'),
                total_packets=data.get('total_packets'),
                avg_latency=data.get('avg_latency'),
                avg_energy=data.get('avg_energy')
            )
            return JsonResponse({"message": "Network status updated successfully."}, status=201)
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON format in request."}, status=400)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)
    else:
        return JsonResponse({"error": "Invalid request method."}, status=405)

def get_network_status_data(request):
    """
    Endpoint to retrieve the latest network status data.
    Returns data for display on the frontend.
    """
    if request.method == 'GET':
        try:
            # Get the latest NetworkStatus object (you might need to adjust ordering)
            latest_status = NetworkStatus.objects.latest('id') # Assuming 'id' is an auto-incrementing field
            status_data = {
                "total_nodes": latest_status.total_nodes,
                "active_nodes": latest_status.active_nodes,
                "total_packets": latest_status.total_packets,
                "avg_latency": latest_status.avg_latency,
                "avg_energy": latest_status.avg_energy
            }
            return JsonResponse(status_data, status=200)
        except NetworkStatus.DoesNotExist:
            logger.warning("No NetworkStatus data available in the database.")
            return JsonResponse({"error": "No network status data available."}, status=404) 
        except Exception as e:
            logger.exception("Error retrieving network status data:")
            return JsonResponse({"error": "Error retrieving network status data."}, status=500)
    else:
        return JsonResponse({"error": "Invalid request method. Use GET."}, status=405)
    
def get_snn_settings(request):
    """
    Endpoint to provide SNN settings to the script.
    Fetches the latest SNNSettings.
    """
    if request.method == 'GET':
        latest_settings = SNNSettings.objects.last()
        if latest_settings:
            settings_data = {
                'learning_rate': latest_settings.learning_rate,
                'synaptic_weight': latest_settings.synaptic_weight,
                'firing_threshold': latest_settings.firing_threshold,
                'reset_potential': latest_settings.reset_potential,
                'tau_pre': latest_settings.tau_pre,
                'tau_post': latest_settings.tau_post,
                'simulation_time': latest_settings.simulation_time,
            }
            return JsonResponse(settings_data, status=200)
        else:
            return JsonResponse({"error": "No SNN settings found."}, status=404)
    else:
        return JsonResponse({"error": "Invalid request method."}, status=405)

@csrf_exempt
def receive_simulation_data(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body.decode('utf-8'))

            routing_path_data = data.get('routing_path_data') # Get routing path data from simulation data

            # --- Save RoutingPath data FIRST ---
            routing_path_instance = None # Initialize to None
            if routing_path_data:
                neighbor_scores_history = routing_path_data.get('neighbor_scores_history', {}) # Get history, default to empty dict if missing

                routing_path_instance = RoutingPath(
                    timestamp=timezone.datetime.now(),
                    stdp_path=json.dumps(routing_path_data.get('selected_path')),
                    stdp_path_score=routing_path_data.get('stdp_path_score'),
                    source_node=0, # Or get from data if source/dest are dynamic
                    destination_node=4, # Or get from data
                    neighbor_scores_history_json=json.dumps(neighbor_scores_history), # NEW: Save neighbor_scores_history as JSON
                )
                print("Received data:", routing_path_instance)
                routing_path_instance.save() # Save RoutingPath instance

                # --- Save PathHistory data (for Dijkstra and STDP paths) ---
                dijkstra_history = PathHistory( # You might want to remove Dijkstra path history saving if you are not using Dijkstra paths anymore in frontend
                    timestamp=timezone.now(),
                    path_type='DIJKSTRA', # Use PATH_TYPE_CHOICES directly
                    path_nodes=routing_path_data.get('dijkstra_path', []), # Use get with default empty list in case dijkstra path is not sent
                    source_node=0,
                    destination_node=4,
                    routing_path=routing_path_instance # Link to the RoutingPath record
                )
                dijkstra_history.save()

                stdp_history = PathHistory(
                    timestamp=timezone.now(),
                    path_type='STDP', # Use PATH_TYPE_CHOICES directly
                    path_nodes=routing_path_data.get('selected_path'),
                    source_node=0,
                    destination_node=4,
                    routing_path=routing_path_instance # Link to the same RoutingPath record
                )
                stdp_history.save()


            # --- Save SimulationDataPoint (existing functionality, now link to RoutingPath) ---
            simulation_data_point = SimulationDataPoint(
                timestamp=timezone.datetime.fromisoformat(data.get('timestamp').replace('Z', '+00:00')),
                simulation_time_ms=data.get('simulation_time_ms'),
                firing_rates_json=json.dumps(data.get('firing_rates')),
                latencies_ms_json=json.dumps(data.get('latencies_ms')),
                energy_levels_json=json.dumps(data.get('energy_levels')),
                congestion_levels_json=json.dumps(data.get('congestion_levels')),
                routing_path=routing_path_instance # Link SimulationDataPoint to RoutingPath
            )
            simulation_data_point.save()

            # --- Save NetworkStatus data --- (No change needed here if you are already saving NetworkStatus)
            status_data = data.get('status_data')
            if status_data:
                network_status = NetworkStatus(
                    timestamp=timezone.datetime.fromisoformat(status_data.get('timestamp').replace('Z', '+00:00')),
                    total_nodes=status_data.get('total_nodes'),
                    active_nodes=status_data.get('active_nodes'),
                    total_packets=status_data.get('total_packets'),
                    avg_latency=status_data.get('avg_latency'),
                    avg_energy=status_data.get('avg_energy')
                )
                network_status.save()


            return JsonResponse({"message": "Simulation and Routing Path data saved successfully!"}, status=201)

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON format in request body"}, status=400)
        except Exception as e:
            return JsonResponse({"error": f"Error saving data: {str(e)}"}, status=400)
    else:
        return JsonResponse({"error": "Only POST requests are allowed"}, status=405)
    

def generate_path_explanation_report(neighbor_scores_history, selected_path_nodes_stdp):
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-pro")

        neighbor_scores_history_json = json.dumps(neighbor_scores_history, indent=2)

        prompt = f"""
        To give context: This is a result of a simulated Spiking Neural Network (SNN) for routing, incorporating biologically plausible mechanisms like adaptive neuron thresholds, STDP (Spike-Timing-Dependent Plasticity), and dynamic routing based on neuron energy levels, congestion, and latency. 
        It fetches SNN settings from a Django API, updates network status and simulation data (including spike trains, firing rates, latencies, energy/congestion levels, and routing path information) to other APIs. 
        The simulation iteratively runs, updating neuron energy and congestion based on spiking activity, with energy recovering over time and consumption increasing with spikes. It calculates dynamic routing metrics (latency, energy, congestion, firing rate) for each neuron and uses these to compute a composite score. 
        Two pathfinding algorithms are implemented: Dijkstra's algorithm with dynamically adjusted edge weights based on these metrics, and a novel STDP-guided Depth-First Search (DFS) algorithm which explores paths based on the composite scores of neighboring neurons.
        Involve the role of other params given:
        ---

        Explain why a path was taken:

        In your explanation, please specifically involve and discuss the role of these parameters:
        - Adaptive neuron thresholds
        - STDP (Spike-Timing-Dependent Plasticity)
        - Neuron energy levels
        - Neuron congestion levels
        - Latency
        - Firing rate
        - Composite score

        Use the following neighbor scores history to help inform your explanation:
        ```json
        {neighbor_scores_history_json}
        ```

        Focus *only* on explaining the path selection and the influence of the parameters. Do not generate any other introductory or concluding text.
        """

        result = llm.invoke(prompt)
        return result.content

    except Exception as e:
        error_message = f"Error generating report from Gemini Pro: {e}"
        logging.error(error_message)
        return error_message # Return error message instead of text
      
def generate_path_explanation_report_sync(simulation_data_point):
    """
    Generates the Gemini path explanation report SYNCHRONOUSLY (blocking request).
    Saves the report to a text file in the file system.
    For demonstration purposes. In production, use Celery.

    Args:
        simulation_data_point (SimulationDataPoint): SimulationDataPoint object.

    Returns:
        str: The generated report text from Gemini Pro, or an error message.
    """
    try:
        routing_path = simulation_data_point.routing_path # Access related RoutingPath object
        if routing_path: # Check if routing_path exists (ForeignKey can be null)
            neighbor_scores_history_json_str = routing_path.neighbor_scores_history_json # Get JSON string from RoutingPath model
            neighbor_scores_history = json.loads(neighbor_scores_history_json_str) if neighbor_scores_history_json_str else {} # Parse JSON string to dict
            stdp_path_str = routing_path.stdp_path # Get stdp_path string from RoutingPath model
            selected_path_nodes_stdp = json.loads(stdp_path_str) if stdp_path_str else [] # Parse JSON string to list of nodes
        else: # Handle case where routing_path is None
            logging.warning(f"No RoutingPath associated with SimulationDataPoint ID: {simulation_data_point.id}. Cannot generate report.")
            return "No routing path data available for report generation." # Return message indicating no data

        report_text = generate_path_explanation_report(neighbor_scores_history, selected_path_nodes_stdp)  # Generate report using Gemini

        if report_text and not report_text.startswith(
                "Error generating report"):  # Check if report generation was successful
            report_file_path = get_report_file_path(simulation_data_point.id)
            try:
                with open(report_file_path, 'w') as report_file:
                    report_file.write(report_text)  # Save report to text file
                logging.info(f"Gemini report saved to file: {report_file_path}")
            except Exception as e_save:
                error_message_save = f"Error saving report to file {report_file_path}: {e_save}"
                logging.error(error_message_save, exc_info=True)
                return error_message_save  # Return error message if saving fails

        return report_text  # Return the generated report text (or error message)

    except Exception as e:
        error_message = f"Error generating report synchronously: {e}"
        logging.error(error_message, exc_info=True)
        return error_message  # Return error message
    
    
####################################################

def dashboard_view(request):
    """
    View to render the dashboard page, fetching NetworkStatus data.
    """
    latest_status = NetworkStatus.objects.last() # Get the latest record
    recent_statuses = NetworkStatus.objects.order_by('-timestamp')[:10]

    print("Value of recent_statuses right before rendering:")
    for status in recent_statuses:
        print(f"  {status}")

    recent_statuses_list = []
    for status in recent_statuses:
        status_dict = { 
            'total_nodes': status.total_nodes,
            'active_nodes': status.active_nodes,
            'total_packets': status.total_packets,
            'avg_latency': status.avg_latency,
            'avg_energy': status.avg_energy,
            'timestamp': status.timestamp.isoformat() # CORRECTLY serialize timestamp to ISO format
        }
        recent_statuses_list.append(status_dict) # CORRECTLY append each status_dict in the loop

    recent_statuses_json = json.dumps(recent_statuses_list) # Serialize the list of dictionaries to JSON

    context = {
        'latest_status': latest_status,
        'recent_statuses_json': recent_statuses_json, # Pass the JSON string in the context
    }
    return render(request, 'main_project/home.html', context)
def historical_routing_view(request):
    """
    View to render the historical routing decisions and performance page.
    """
    historical_routing_decisions = PathHistory.objects.order_by('-timestamp')[:50] 
    recent_statuses = NetworkStatus.objects.order_by('-timestamp')[:50]
    neuron_metrics = NeuronMetrics.objects.order_by('-timestamp')[:50]

    context = {
        'historical_routing_decisions': historical_routing_decisions,
        'recent_statuses': recent_statuses,
        'neuron_metrics': neuron_metrics,
    }
    return render(request, 'main_project/historical_routing.html', context)

from django.shortcuts import render, get_object_or_404
from .models import SimulationDataPoint
import json # To load JSON strings from DB

def simulation_data_point_list(request):
    data_points = SimulationDataPoint.objects.all().order_by('-timestamp') # Order by timestamp, newest first
    return render(request, 'main_project/simulation_data_list.html', {'data_points': data_points})

def simulation_data_point_detail(request, pk):
    """View to display details of a specific SimulationDataPoint record."""
    data_point = get_object_or_404(SimulationDataPoint, pk=pk)

    firing_rates = json.loads(data_point.firing_rates_json)
    latencies_ms = json.loads(data_point.latencies_ms_json)
    energy_levels = json.loads(data_point.energy_levels_json)
    congestion_levels = json.loads(data_point.congestion_levels_json)
    routing_path = data_point.routing_path

    context = {
        'data_point': data_point,
        'firing_rates': firing_rates,
        'latencies_ms': latencies_ms,
        'energy_levels': energy_levels,
        'congestion_levels': congestion_levels,
        'routing_path': routing_path,
    }
    return render(request, 'main_project/simulation_data_detail.html', context)

from django.shortcuts import render
from django.conf import settings
import os
import logging

def view_simulation_log(request):
    """
    View to display the content of the simulation.log file, 
    truncated to the last 200 lines and formatted for display.
    """
    log_file_path = os.path.join(settings.BASE_DIR, 'scripts', 'simulation.log')

    log_lines_raw = [] # List to hold raw log lines
    try:
        with open(log_file_path, 'r') as logfile:
            log_lines_raw = logfile.readlines() # Read all lines into a list
    except FileNotFoundError:
        log_content = "Log file not found."
        logging.warning(f"Log file not found at: {log_file_path}")
    except Exception as e:
        log_content = f"Error reading log file: {e}"
        logging.error(f"Error reading log file at {log_file_path}: {e}")

    # Truncate to last 200 lines
    log_lines_raw = log_lines_raw[-150:]

    log_lines_parsed = [] # List to hold parsed log lines as dictionaries
    for line in log_lines_raw:
        try:
            parts = line.strip().split(' - ', 2) # Split into timestamp, levelname, message (maxsplit=2)
            if len(parts) == 3: # Ensure line has all 3 parts
                asctime, levelname, message = parts
                log_lines_parsed.append({
                    'asctime': asctime,
                    'levelname': levelname,
                    'message': message,
                })
            else: # Handle lines that don't match expected format (optional logging/handling)
                logging.warning(f"Skipping unparseable log line: {line.strip()}") # Log unparseable lines
                continue # Or append raw line as is, or handle differently

        except ValueError: # Handle potential split errors (less likely with maxsplit=2, but good practice)
            logging.warning(f"Error parsing log line: {line.strip()}")
            continue # Or append raw line as is, or handle differently


    context = {
        'log_lines': log_lines_parsed, 
    }
    return render(request, 'main_project/simulation_log.html', context)

from django.contrib import messages  # Import messages framework

def snn_control_panel_view(request):

    latest_settings = SNNSettings.objects.last()
    if not latest_settings:
        latest_settings = SNNSettings.objects.create()

    if request.method == 'POST':
        form_data = request.POST
        latest_settings.learning_rate = float(form_data.get('learning_rate', latest_settings.learning_rate))
        latest_settings.synaptic_weight = float(form_data.get('synaptic_weight', latest_settings.synaptic_weight))
        latest_settings.firing_threshold = float(form_data.get('firing_threshold', latest_settings.firing_threshold))
        latest_settings.reset_potential = float(form_data.get('reset_potential', latest_settings.reset_potential))
        latest_settings.tau_pre = float(form_data.get('tau_pre', latest_settings.tau_pre))
        latest_settings.tau_post = float(form_data.get('tau_post', latest_settings.tau_post))
        latest_settings.simulation_time = float(form_data.get('simulation_time', latest_settings.simulation_time))
        latest_settings.save()
        messages.success(request, 'SNN Settings updated successfully!') # Success message

        return redirect('snn_control_panel') # Redirect to the control panel page after POST

    context = {
        'snn_settings': latest_settings,
    }
    return render(request, 'main_project/snn_control_panel.html', context)

def view_simulation_report(request, simulation_id):

    simulation_data = get_object_or_404(SimulationDataPoint, id=simulation_id)
    report_file_path = get_report_file_path(simulation_id)  # Get file path for this simulation_id

    report_text = None  # Initialize report_text as None

    if os.path.exists(report_file_path):  # Check if report file exists
        try:
            with open(report_file_path, 'r') as report_file:
                report_text = report_file.read()  # Read report from file
            logging.info(f"Fetched existing Gemini report from file for SimulationData ID: {simulation_id}")
        except Exception as e:
            report_text = f"Error reading report file: {e}"
            logging.error(f"Error reading report file for SimulationData ID {simulation_id}: {e}", exc_info=True)

    if not report_text:  # If report file doesn't exist or reading failed, generate it
        # --- Corrected function call - pass only simulation_data ---
        report_text = generate_path_explanation_report_sync(simulation_data) # Corrected line: Passing only simulation_data
        if report_text and not report_text.startswith(
                "Error generating report"):  # Check if report generation was successful
            logging.info(f"Gemini report generated and saved new Gemini report to file for SimulationData ID: {simulation_id}")
        else:
            logging.warning(f"Failed to generate Gemini report for SimulationData ID: {simulation_id}")

    context = {
        'simulation_data': simulation_data,
        'report_text': report_text,
    }
    return render(request, 'main_project/simulation_report.html', context)


import requests
from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings

PINATA_API_KEY = "897bc0dbc9c242fe0c35"
PINATA_SECRET_API_KEY = "0c7ce37e46f40b32b70eee0f85ffcb43af2704dcb2af60e88d3d81e70c9fb436"
PINATA_URL = "https://api.pinata.cloud/pinning/pinFileToIPFS"

def upload_to_pinata(file):
    headers = {
        "pinata_api_key": PINATA_API_KEY,
        "pinata_secret_api_key": PINATA_SECRET_API_KEY,
    }
    files = {
        "file": (file.name, file.read())
    }
    response = requests.post(PINATA_URL, files=files, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.text}

def ipfs_upload_view(request):
    if request.method == "POST":
        uploaded_file = request.FILES.get("file")
        if uploaded_file:
            result = upload_to_pinata(uploaded_file)
            return render(request, "main_project/upload_result.html", {"result": result})
        else:
            return render(request, "main_project/upload_form.html", {"error": "No file uploaded."})
    return render(request, "main_project/upload_form.html")

####RENDER GRAPH
import json
from django.shortcuts import render, get_object_or_404
from django.http import Http404
from django.core.serializers.json import DjangoJSONEncoder # Handles dates/times if needed
from .models import RoutingPath

# (Keep the parse_path_string helper function from the Mermaid example - it's useful)
def parse_path_string(path_str):
    """ Parses various path string formats into a list of node strings. """
    nodes = []
    if not path_str: return nodes
    try:
        parsed_nodes = json.loads(path_str)
        if isinstance(parsed_nodes, list):
            nodes = [str(n).strip() for n in parsed_nodes if str(n).strip()]
            return nodes
    except (json.JSONDecodeError, TypeError): pass
    path_str_cleaned = path_str.strip()
    if not path_str_cleaned: return nodes
    if ' -> ' in path_str_cleaned: nodes = [n.strip() for n in path_str_cleaned.split(' -> ') if n.strip()]
    elif ',' in path_str_cleaned: nodes = [n.strip() for n in path_str_cleaned.split(',') if n.strip()]
    elif ' ' in path_str_cleaned: nodes = [n.strip() for n in path_str_cleaned.split() if n.strip()]
    else: nodes = [path_str_cleaned]
    return nodes


def data_point_detail(request, pk):
    try:
        routing_path = get_object_or_404(RoutingPath, pk=pk)
    except (RoutingPath.DoesNotExist, ValueError):
        routing_path = None
    except Exception as e:
        print(f"Error fetching routing path: {e}")
        routing_path = None

    vis_graph_data = None # Data structure for Vis.js

    if routing_path and routing_path.stdp_path:
        try:
            nodes_in_path = parse_path_string(routing_path.stdp_path)

            if nodes_in_path:
                vis_nodes = []
                vis_edges = []

                # Create Vis.js nodes
                for i, node_id in enumerate(nodes_in_path):
                    node_data = {
                        'id': node_id,       # Use the node ID from path
                        'label': str(node_id) # Display the node ID as label
                    }
                    # Highlight start and end nodes visually
                    if i == 0:
                        node_data['color'] = {'background': '#d4edda', 'border': '#155724'}
                        node_data['font'] = {'color': '#155724'}
                    elif i == len(nodes_in_path) - 1:
                         node_data['color'] = {'background': '#f8d7da', 'border': '#721c24'}
                         node_data['font'] = {'color': '#721c24'}
                    vis_nodes.append(node_data)

                # Create Vis.js edges (connections)
                for i in range(len(nodes_in_path) - 1):
                    vis_edges.append({
                        'from': nodes_in_path[i],
                        'to': nodes_in_path[i+1],
                        'arrows': 'to' # Show direction
                    })

                # Package data for the template
                vis_graph_data = {
                    'nodes': vis_nodes,
                    'edges': vis_edges
                }

        except Exception as e:
            print(f"Error preparing data for Vis.js: {e}")
            vis_graph_data = None

    context = {
        'routing_path': routing_path,
        # Pass the Vis.js data as JSON
        'vis_graph_data_json': json.dumps(vis_graph_data, cls=DjangoJSONEncoder) if vis_graph_data else None,
        # Keep other context variables if needed
    }
    return render(request, 'graph.html', context)

##NEW GRAPH
# Add near other imports at the top of views.py
from django.core.serializers.json import DjangoJSONEncoder
from django.http import Http404, JsonResponse # Import JsonResponse if not already there
from .models import SimulationDataPoint, RoutingPath # Make sure models are imported
import json
import logging

logger = logging.getLogger(__name__) # Ensure logger is defined

# --- NEW VIEW ---
def simulation_data_visualizer_view(request, pk):
    """
    View to render the graph visualization for a specific SimulationDataPoint ID (pk).
    Data is loaded once and passed to the template.
    """
    try:
        # 1. Fetch the specific SimulationDataPoint
        data_point = get_object_or_404(SimulationDataPoint, pk=pk)
        logger.info(f"Fetching visualization data for SimulationDataPoint {pk}")

        # 2. Initialize the payload dictionary for JavaScript
        payload = {
            "timestamp": data_point.timestamp.isoformat() if data_point.timestamp else None,
            "simulation_time_ms": data_point.simulation_time_ms,
            "firing_rates": [],
            "latencies_ms": [],
            "energy_levels": [],
            "congestion_levels": [],
            "cycle_number": getattr(data_point, 'cycle_number', None), # If you add cycle_number later
            "routing_path_data": {},
            "detailed_node_scores": {}, # Initialize as empty dict
        }

        # 3. Safely parse basic per-node data lists from SimulationDataPoint
        try:
            payload["firing_rates"] = json.loads(data_point.firing_rates_json or '[]')
            payload["latencies_ms"] = json.loads(data_point.latencies_ms_json or '[]')
            payload["energy_levels"] = json.loads(data_point.energy_levels_json or '[]')
            payload["congestion_levels"] = json.loads(data_point.congestion_levels_json or '[]')
            logger.debug(f"Parsed basic node lists for {pk}. N={len(payload['firing_rates'])}")
        except json.JSONDecodeError as e_parse:
            logger.error(f"Failed to parse base JSON data for SimulationDataPoint {pk}: {e_parse}")
            # Decide how to handle: raise error, return error page, or continue with empty lists?
            # Let's continue with empty lists for now, JS should handle it.
            pass
        except Exception as e_base:
            logger.error(f"Error loading base data for SimulationDataPoint {pk}: {e_base}")
            pass # Continue with empty lists


        # 4. Get related RoutingPath data and reconstruct detailed scores
        routing_path = data_point.routing_path
        if routing_path:
            logger.info(f"Found related RoutingPath {routing_path.pk}")
            stdp_path_list = json.loads(routing_path.stdp_path or '[]')
            neighbor_scores_history = json.loads(routing_path.neighbor_scores_history_json or '{}')

            # *** RECONSTRUCTION of detailed_node_scores ***
            reconstructed_detailed_scores = {}
            if isinstance(neighbor_scores_history, dict):
                for source_node_str, neighbors in neighbor_scores_history.items():
                    if isinstance(neighbors, dict):
                        for neighbor_idx_str, score_details in neighbors.items():
                            # Store the first instance found for each neighbor ID
                            if neighbor_idx_str not in reconstructed_detailed_scores and isinstance(score_details, dict):
                                reconstructed_detailed_scores[neighbor_idx_str] = score_details
                logger.info(f"Reconstructed detailed scores for {len(reconstructed_detailed_scores)} nodes from history.")
            else:
                logger.warning(f"Neighbor scores history for RoutingPath {routing_path.pk} is not a valid dictionary.")


            payload["routing_path_data"] = {
                "source_node": routing_path.source_node,
                "destination_node": routing_path.destination_node,
                "dijkstra_path": [], # Add logic if Dijkstra path is stored elsewhere
                "dijkstra_cost": None, # Add logic if cost is stored
                "selected_path": stdp_path_list, # Heuristic path
                "stdp_path_score": routing_path.stdp_path_score,
                # "neighbor_scores_history": neighbor_scores_history # Optionally include if JS needs it
            }
            # Use the reconstructed scores
            payload["detailed_node_scores"] = reconstructed_detailed_scores

        else:
            logger.warning(f"No RoutingPath linked to SimulationDataPoint {pk}. Visualization may lack path/score details.")
            # Provide empty defaults for routing data if no link exists
            payload["routing_path_data"] = { "source_node": None, "destination_node": None, "dijkstra_path": [], "dijkstra_cost": None, "selected_path": [], "stdp_path_score": None }
            payload["detailed_node_scores"] = {}


        # 5. Pass data to the template
        context = {
            'simulation_data_point_pk': pk,
            'initial_simulation_data_json': json.dumps(payload, cls=DjangoJSONEncoder),
        }
        # Render the graph.html template, passing the context
        return render(request, 'graph.html', context)

    except SimulationDataPoint.DoesNotExist:
        logger.warning(f"SimulationDataPoint with pk={pk} not found.")
        raise Http404("Simulation Data Point not found.")
    except Exception as e:
        logger.exception(f"Unexpected error loading visualizer view for SimulationDataPoint {pk}: {e}")
        raise Http404(f"An error occurred loading the visualization data for Simulation {pk}.")

    """
    API endpoint to fetch the most recent simulation data record.
    RECONSTRUCTS detailed_node_scores from neighbor history.
    Used by the polling version of graph.html (if needed).
    """
    try:
        latest_data_point = SimulationDataPoint.objects.order_by('-timestamp').first()
        if not latest_data_point:
            return JsonResponse({"error": "No simulation data found."}, status=404)

        logger.info(f"API: Fetching latest data (SimulationDataPoint {latest_data_point.pk})")

        payload = {
            "timestamp": latest_data_point.timestamp.isoformat() if latest_data_point.timestamp else None,
            "simulation_time_ms": latest_data_point.simulation_time_ms,
            "firing_rates": json.loads(latest_data_point.firing_rates_json or '[]'),
            "latencies_ms": json.loads(latest_data_point.latencies_ms_json or '[]'),
            "energy_levels": json.loads(latest_data_point.energy_levels_json or '[]'),
            "congestion_levels": json.loads(latest_data_point.congestion_levels_json or '[]'),
            "cycle_number": getattr(latest_data_point, 'cycle_number', None),
            "routing_path_data": {},
            "detailed_node_scores": {},
        }

        routing_path = latest_data_point.routing_path
        if routing_path:
            logger.info(f"API: Found related RoutingPath {routing_path.pk}")
            stdp_path_list = json.loads(routing_path.stdp_path or '[]')
            neighbor_scores_history = json.loads(routing_path.neighbor_scores_history_json or '{}')

            reconstructed_detailed_scores = {}
            if isinstance(neighbor_scores_history, dict):
                 for source_node_str, neighbors in neighbor_scores_history.items():
                     if isinstance(neighbors, dict):
                         for neighbor_idx_str, score_details in neighbors.items():
                             if neighbor_idx_str not in reconstructed_detailed_scores and isinstance(score_details, dict):
                                 reconstructed_detailed_scores[neighbor_idx_str] = score_details
                 logger.info(f"API: Reconstructed detailed scores for {len(reconstructed_detailed_scores)} nodes.")
            else:
                 logger.warning(f"API: Neighbor scores history for RoutingPath {routing_path.pk} is not a valid dictionary.")


            payload["routing_path_data"] = {
                "source_node": routing_path.source_node,
                "destination_node": routing_path.destination_node,
                "dijkstra_path": [],
                "dijkstra_cost": None,
                "selected_path": stdp_path_list,
                "stdp_path_score": routing_path.stdp_path_score,
                # "neighbor_scores_history": neighbor_scores_history
            }
            payload["detailed_node_scores"] = reconstructed_detailed_scores
        else:
            logger.warning(f"API: No RoutingPath linked to latest SimulationDataPoint {latest_data_point.pk}.")
            payload["routing_path_data"] = { "source_node": None, "destination_node": None, "dijkstra_path": [], "dijkstra_cost": None, "selected_path": [], "stdp_path_score": None }
            payload["detailed_node_scores"] = {}

        return JsonResponse(payload)

    except Exception as e:
        logger.exception(f"API: Error fetching latest simulation data: {e}")
        return JsonResponse({"error": "An internal error occurred fetching data."}, status=500)