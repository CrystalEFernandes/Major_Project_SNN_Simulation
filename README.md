# Optimized Spiking Neural Network Architecture for Quality Routing in IoT-Assisted Wireless Sensor Networks

This repository contains the project code and resources for the Bachelor of Engineering project titled "Optimized Spiking Neural Network Architecture For Quality Routing In IoT Assisted Wireless Sensor Networks". The project focuses on enhancing routing performance in IoT-assisted Wireless Sensor Networks (WSNs) by integrating Spiking Neural Networks (SNNs) for adaptive routing decisions and Blockchain technology for data security and integrity.

**Authors:**
*   Crystal Fernandes (9539)
*   Emmanuel Gudinho (9609)
*   Leslie D'silva (9599)
*   Lisa Gonsalves (9607)

**Guide:** Prof. Prachi Patil (Assistant Professor)

**Institution:** Fr. Conceicao Rodrigues College of Engineering, Bandra (W), Mumbai - 400050 (University of Mumbai)

---

## Table of Contents

*   [Overview](#overview)
*   [Problem Statement](#problem-statement)
*   [Proposed Solution](#proposed-solution)
*   [Key Features](#key-features)
*   [System Architecture](#system-architecture)
*   [Technologies Used](#technologies-used)
*   [Modules](#modules)
*   [Results Highlights](#results-highlights)
*   [Setup and Installation](#setup-and-installation)
*   [Acknowledgments](#acknowledgments)

---

## Overview

The rapid growth of IoT applications (smart cities, healthcare, environmental monitoring) relies heavily on Wireless Sensor Networks (WSNs). However, ensuring secure, efficient, and reliable data routing in these often resource-constrained networks is a significant challenge. Traditional routing protocols struggle with dynamic network conditions and energy limitations. This project proposes a novel architecture combining the strengths of:

1.  **Spiking Neural Networks (SNNs):** Biologically inspired neural networks that process information using spikes, enabling adaptive, energy-efficient routing decisions based on real-time network conditions like latency, node energy, and congestion.
2.  **Blockchain Technology:** Provides a decentralized, immutable ledger to ensure secure, tamper-proof communication and data integrity, particularly for storing routing information or data identifiers.
3.  **IPFS (InterPlanetary File System):** Used for decentralized storage of sensor data, enhancing resilience and accessibility.

The goal is to create a robust, scalable, and secure routing solution optimized for quality and efficiency in IoT-assisted WSNs.

## Problem Statement

Traditional WSN routing protocols and even some modern machine learning integrations face several challenges:
*   **High Computational/Energy Overheads:** Resource-intensive processing unsuitable for constrained WSN nodes.
*   **Network Inefficiencies:** Potential for delayed transmission, increased latency, and uneven workload distribution.
*   **Reduced Network Lifespan:** High energy consumption shortens node and overall network lifetime.
*   **Security Vulnerabilities:** Inadequate handling of malicious nodes, susceptibility to data tampering, unauthorized access, and compromised routing.
*   **Static Routing:** Inability to adapt efficiently to dynamic network topology and link quality changes.

## Proposed Solution

This project implements an SNN-based dynamic routing system enhanced with blockchain for security.
1.  Sensor data (e.g., camera images) is collected and stored decentrally on IPFS.
2.  The unique Content Identifier (CID) from IPFS is recorded on a blockchain for tamper-proof verification.
3.  An SNN, specifically using the Leaky Integrate-and-Fire (LIF) model with adaptive thresholds and Spike-Timing-Dependent Plasticity (STDP) for learning, analyzes network conditions (latency, energy, congestion, firing rate) of potential routing nodes.
4.  A heuristic routing algorithm (modified Dijkstra's or Best-First Search) uses a composite score derived from SNN metrics to determine the optimal path for data packets (or CIDs).
5.  The SNN adapts synaptic weights based on path performance, continuously optimizing routing decisions.
6.  Data is routed securely and efficiently towards a central server or destination node for processing.

## Key Features

*   **Adaptive Routing:** SNN dynamically adjusts routing paths based on real-time network conditions.
*   **Energy Efficiency:** SNNs, especially when potentially implemented on neuromorphic hardware, offer significant energy savings compared to traditional methods.
*   **Enhanced Security:** Blockchain provides a tamper-proof record for data CIDs and potentially routing logs.
*   **Data Integrity:** IPFS ensures decentralized and resilient data storage.
*   **Optimized Path Selection:** Considers multiple metrics (latency, energy, congestion, reliability) using a composite score for intelligent routing.
*   **Dynamic Learning:** STDP allows the network to learn and reinforce efficient and reliable paths over time.
*   **Improved Reliability:** Demonstrates high Packet Delivery Ratios (PDR) and Route Stability, especially in specific topologies.

## System Architecture

The system follows this general flow:

1.  **Data Collection & Storage:** Sensors capture data -> Data encrypted -> Stored on IPFS -> Returns unique CID.
2.  **Blockchain Record:** The CID is recorded on the Blockchain.
3.  **SNN Routing:**
    *   Source node initiates routing request.
    *   SNN evaluates potential next-hop nodes based on learned weights and real-time metrics (energy, latency, congestion, firing rate).
    *   A composite score is calculated for each potential path/node.
    *   The path with the best score is selected using a heuristic search.
    *   Weights are updated via STDP based on routing success/failure.
4.  **Data Transmission:** Encrypted data/CID travels through the selected path of router nodes.
5.  **Central Processing:** Destination/Server receives data -> Decrypts -> Processes/Analyzes.

![proposed_system](https://github.com/user-attachments/assets/6564b994-b22f-4bcc-84e9-09fafc189cdd)

## Technologies Used

*   **Spiking Neural Network Simulation:**
    *   Python
    *   Brian2 / NEST (SNN Simulation Libraries)
    *   NetworkX (Graph representation and analysis)
    *   NumPy, Matplotlib (Numerical operations, Plotting)
*   **Decentralized Storage:** IPFS
*   **Blockchain:** Conceptual implementation 
*   **Routing Algorithm:** Heuristic Best-First Search
*   **Frontend (Optional/Visualization):** Html, Css, JS
*   **Development Environment:** Google Colab, Local Python Environment, Django

## Modules

1.  **Data Collection/Handling:** Simulating sensor input, interacting with IPFS (uploading data, retrieving CIDs).
2.  **Blockchain Interaction:** Recording CIDs, potentially storing/querying routing paths.
3.  **SNN Core (LIF Model, STDP):** Defining neuron equations, synapse models, plasticity rules.
4.  **Network Simulation:** Running the SNN simulation using Brian2.
5.  **Routing Metric Calculation:** Computing latency, energy consumption, congestion, firing rates, synaptic weights, and the composite score.
6.  **Routing Algorithm:** Implementing the pathfinding logic.
7.  **Visualization:** Plotting network graphs, metrics, simulation results.
8.  **Main Simulation Orchestrator:** Script to initialize, run, and manage the different components.

## Results Highlights

Comparative analysis against OSPF and BGP across Tree, Mesh, and Star topologies revealed:

*   **Packet Delivery Ratio (PDR):** SNN achieved perfect PDR (1.0) in Tree/Star, outperforming OSPF/BGP (0.80-0.89).
*   **Energy Consumption:** SNN significantly lower, especially in Star (3,417 mJ vs ~20,270 mJ) and Tree (19,093 mJ vs ~34,000 mJ). Higher in Mesh due to simulation complexity.
*   **Route Stability:** SNN achieved perfect stability (1) in Tree/Star, whereas OSPF/BGP were unstable (0 in Mesh/Star).
*   **Path Length:** SNN found shorter paths in Tree (197 vs 333-357 hops) and Star (44 vs 249-251 hops).
*   **Trade-offs:** SNN generally has higher execution/convergence times due to the complexity of the simulation but offers superior reliability and efficiency, particularly for specific topologies.

*(Refer to Tables 6.1 - 6.5 in the report for detailed metrics)*

## Setup and Installation

1.  **Prerequisites:**
    *   Python (e.g., 3.8+)
    *   `pip`
    *   Git
    *   A Pinata Account ([pinata.cloud](https://pinata.cloud/)) for persistent IPFS pinning.

2.  **Clone the Repository:**
    ```bash
    git clone (https://github.com/CrystalEFernandes/Major_Project_SNN_Simulation)
    ```

3.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    
4.  **API Key Configuration (Environment Variables):**
    This project requires API keys. Store these securely as environment variables.

    ```dotenv
    # .env file
    PINATA_API_KEY="YOUR_PINATA_API_KEY"
    PINATA_API_SECRET_KEY="YOUR_PINATA_API_SECRET_KEY"
    # GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
    ```

5.  **Django Setup:**
    *   **Apply Database Migrations:**
        ```bash
        python manage.py migrate
        ```
    *   **(Optional) Create a Superuser:** To access the Django admin interface.
        ```bash
        python manage.py createsuperuser
        ```
---

## Acknowledgments

*   We express sincere gratitude to our guide, **Prof. Prachi Patil**, for her invaluable technical guidance and suggestions.
*   We thank Dr. Sujata Deshmukh (Head of Computer Engineering Dept.), Dr. S.S. Rathod (Principal), and the management of C.R.C.E., Mumbai, for providing the necessary infrastructure and encouragement.
*   Thanks to all non-teaching staff for their support.
*   This work is dedicated to our families for their motivation and support.
