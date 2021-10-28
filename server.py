import flwr as fl
import sys
import numpy as np

# Start Flower server for three rounds of federated learning
fl.server.start_server(
        server_address = 'localhost:5000', 
        config={"num_rounds": 2}
)