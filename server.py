import flwr as fl
from flwr.server.strategy import FedAvg

# Start Flower server for three rounds of federated learning
fl.server.start_server(
    server_address="localhost:5000",
    config={"num_rounds": 3},
    strategy=FedAvg(min_fit_clients=3, min_available_clients=3),
)
