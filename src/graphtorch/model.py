import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class InteractionNetwork(nn.Module):
    def __init__(self, node_features_track, node_features_hit, edge_features, hidden_size):
        super().__init__()
        
        # 1. Encoders: Embed raw inputs into latent space
        self.track_encoder = nn.Sequential(
            nn.Linear(node_features_track, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.hit_encoder = nn.Sequential(
            nn.Linear(node_features_hit, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # 2. Processor: The core interaction/graph logic
        self.processor = RelationalProcessor(hidden_size)
        
        # 3. Edge Decoder: Classifies edges based on processed latent features
        # Input size is 3 * hidden_size because we concat (track + hit + edge)
        self.edge_classifier = nn.Sequential(
            nn.Linear(3 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1) # Logits
        )

    def forward(self, x_track, x_hit, edge_index, edge_attr):
        # 1. Encode
        h_track = self.track_encoder(x_track)
        h_hit = self.hit_encoder(x_hit)
        h_edge = self.edge_encoder(edge_attr)
        
        # 2. Process
        # Currently this returns the concatenated features (Deep Set style)
        # In future GNN updates, this will return updated latent features after message passing
        latent_edge_features = self.processor(h_track, h_hit, h_edge, edge_index)
        
        # 3. Decode
        edge_logits = self.edge_classifier(latent_edge_features)
        
        return edge_logits

class RelationalProcessor(MessagePassing):
    """
    Current: 'Deep Set' / 'Relational Network' (No Aggregation)
    Future:  Upgrade this class to perform Message Passing (GNN)
    """
    def __init__(self, hidden_size):
        super().__init__(aggr='add') # Aggr will be used in future updates
        self.hidden_size = hidden_size

    def forward(self, h_track, h_hit, h_edge, edge_index):
        # Unpack indices
        src, dst = edge_index
        
        # In a full GNN, we would call self.propagate() here.
        # For now, we manually construct the 'message' or 'relation'
        # by concatenating the features of the connected nodes and the edge.
        
        # Construct edge representations: [h_track[src] || h_hit[dst] || h_edge]
        # Shape: [num_edges, 3 * hidden_size]
        edge_representation = torch.cat([h_track[src], h_hit[dst], h_edge], dim=1)
        
        return edge_representation

# Basic model wrapper for HeteroData
class GraphTORCHModel(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        # Track: [x, y, dx, dy, p, t0] -> 6
        # Hit: [x, y, t] -> 3
        # Edge: [delta_t] -> 1
        self.model = InteractionNetwork(6, 3, 1, hidden_size)
        
    def forward(self, data):
        # Unpack HeteroData
        x_track = data['track'].x
        x_hit = data['hit'].x
        edge_index = data['track', 'to', 'hit'].edge_index
        edge_attr = data['track', 'to', 'hit'].edge_attr
        
        return self.model(x_track, x_hit, edge_index, edge_attr)
