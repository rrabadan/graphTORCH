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
            nn.Linear(hidden_size, 2) # Logits
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

from torch_geometric.utils import scatter

class RelationalProcessor(nn.Module):
    """
    Bi-Directional Message Passing (Track <-> Hit).
    Allows Hits to communicate via the Track hub.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # --- PASS 1: Track -> Hit ---
        # Msg = MLP(h_track || h_edge)
        self.msg_t2h = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        # Update Hit
        self.update_hit = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # --- PASS 2: Hit -> Track (Backward) ---
        # Msg = MLP(h_hit || h_edge)
        self.msg_h2t = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        # Update Track
        self.update_track = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, h_track, h_hit, h_edge, edge_index):
        src, dst = edge_index
        
        # --- STEP 1: Hit -> Track (Backward Flow first?) ---
        # Usually good to let Tracks aggregate info from their candidates first.
        # Flow: Hit(dst) -> Track(src)
        
        # Msg construction
        msg_h2t = self.msg_h2t(torch.cat([h_hit[dst], h_edge], dim=1))
        
        # Aggregate at Source (Tracks)
        # Note: edge_index[0] is src (tracks).
        aggr_h2t = scatter(msg_h2t, src, dim=0, dim_size=h_track.size(0), reduce='add')
        
        # Update Tracks
        h_track_updated = self.update_track(torch.cat([h_track, aggr_h2t], dim=1))
        
        
        # --- STEP 2: Track -> Hit (Forward Flow) ---
        # Now use the UPDATED track features to check overlap/compatibility
        # Flow: Track(src) -> Hit(dst)
        
        # Msg construction (Using UPDATED tracks)
        msg_t2h = self.msg_t2h(torch.cat([h_track_updated[src], h_edge], dim=1))
        
        # Aggregate at Target (Hits)
        aggr_t2h = scatter(msg_t2h, dst, dim=0, dim_size=h_hit.size(0), reduce='add')
        
        # Update Hits
        h_hit_updated = self.update_hit(torch.cat([h_hit, aggr_t2h], dim=1))
        
        
        # --- STEP 3: Classify Edges ---
        # Use UPDATED Track AND UPDATED Hit features
        edge_representation = torch.cat([h_track_updated[src], h_hit_updated[dst], h_edge], dim=1)
        
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
