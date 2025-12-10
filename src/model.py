
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class InteractionNetwork(nn.Module):
    def __init__(self, node_features_track, node_features_hit, edge_features, hidden_size):
        super().__init__()
        
        # Encoders
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
        
        # Message Passing
        self.processor = RelationalProcessor(hidden_size)
        
        # Edge Decoder (Classifier)
        # Takes (track_emb, hit_emb, edge_emb) -> score
        self.edge_classifier = nn.Sequential(
            nn.Linear(3 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1) # Logits
        )

    def forward(self, x_track, x_hit, edge_index, edge_attr):
        # Encode inputs
        h_track = self.track_encoder(x_track)
        h_hit = self.hit_encoder(x_hit)
        h_edge = self.edge_encoder(edge_attr)
        
        # Process (Interaction Network step)
        # We want to update edges based on nodes and previous edge state
        # Then we might update nodes? The goal is edge classification.
        # Simple interaction: Update edge_attr using incident nodes.
        
        # For simple edge classification we can just compute:
        # edge_prime = MLP(h_track[src], h_hit[dst], h_edge)
        # But let's use the processor to be safe (though full MPNN might be overkill if we just want link prediction based on static properties? 
        # Actually user said "perform edge classification", often this implies using the graph structure.
        # Let's do:
        # 1. Update edges based on connected nodes.
        # 2. Update nodes based on connected edges (optional, but good for context).
        # 3. Classify edges.
        
        # Let's stick to a simple relational block for now:
        # Edge update: e' = phi_e(h_Source, h_Target, e)
        
        src, dst = edge_index
        
        input_classifier = torch.cat([h_track[src], h_hit[dst], h_edge], dim=1)
        
        edge_logits = self.edge_classifier(input_classifier)
        
        return edge_logits

class RelationalProcessor(MessagePassing):
    # This could be used if we wanted multiple layers of MPNN.
    # For now, the implementation above effectively does one pass which is often enough if features are good.
    # If the user wants a deeper GNN, we can expand.
    # Let's keep it simple: 1-step inference as implemented in InteractionNetwork forward.
    def __init__(self, hidden_size):
        super().__init__(aggr='add')

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
