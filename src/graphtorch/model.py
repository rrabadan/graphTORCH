import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter

class InteractionNetwork(nn.Module):
    def __init__(self, node_features_track, node_features_hit, edge_features, hidden_size):
        super().__init__()
        
        # --- 1. Encoders ---
        # These project Physics quantities into the Latent Space.
        # node_features_track = 6 (x, y, tx, ty, p, t0)
        # node_features_hit = 3 (x, y, t)
        # edge_features = 3 (dt, dx, dy)
        
        self.track_encoder = nn.Sequential(
            nn.Linear(node_features_track, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()  # Added extra non-linearity
        )
        
        self.hit_encoder = nn.Sequential(
            nn.Linear(node_features_hit, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # --- 2. Processor ---
        self.processor = RelationalProcessor(hidden_size)
        
        # --- 3. Edge Decoder ---
        # Classifies the edge based on the final latent representation.
        # Input: [Updated Track (Hidden) + Updated Hit (Hidden) + Edge (Hidden)]
        self.edge_classifier = nn.Sequential(
            nn.Linear(3 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1) # Output 1 score (Logit) for BCEWithLogitsLoss
        )

    def forward(self, x_track, x_hit, edge_index, edge_attr):
        # 1. Encode Raw Physics Data -> Latent Space
        h_track = self.track_encoder(x_track)
        h_hit = self.hit_encoder(x_hit)
        h_edge = self.edge_encoder(edge_attr)
        
        # 2. Process Graph (Message Passing)
        # Returns: Tensor of shape [num_edges, 3 * hidden_size]
        edge_latent_features = self.processor(h_track, h_hit, h_edge, edge_index)
        
        # 3. Decode -> Classification Score
        edge_logits = self.edge_classifier(edge_latent_features)
        
        return edge_logits


class RelationalProcessor(nn.Module):
    """
    Interaction Network Processor.
    'Pair-Aware' Message Passing.
    
    The network computes a latent 'Interaction Representation' for every 
    Track-Hit pair BEFORE aggregation. This aims at allowing the model to learn 
    the optical transfer function (Track -> Optical Path -> Hit) explicitly 
    at the edge level.
    """
    def __init__(self, hidden_size):
        super().__init__()
        
        # 1. The "Interaction" Learner (Edge Block)
        # Input: Track + Hit + Edge
        # Logic: f(Track, Hit, Edge) -> Latent Interaction
        # This MLP aims at learning: "Is this specific Hit consistent with this specific Track?"
        self.edge_updater = nn.Sequential(
            nn.Linear(3 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size) 
        )
        
        # 2. Track Updater (Node Block)
        # Aggregates interactions to update Track belief
        self.track_updater = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size), # [Old Track || Aggr Message]
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # 3. Hit Updater (Node Block)
        # Aggregates interactions to update Hit belief
        self.hit_updater = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size), # [Old Hit || Aggr Message]
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, h_track, h_hit, h_edge, edge_index):
        src, dst = edge_index
        
        # --- STEP 1: Compute Interactions (The "Optical" Check) ---
        # Concatenate features for every pair: [Track_i || Hit_j || Edge_ij]
        # This places the Track kinematics and Hit position in the same vector
        # allowing the MLP to learn the non-linear relationship between them.
        pair_features = torch.cat([h_track[src], h_hit[dst], h_edge], dim=1)
        
        # Compute latent edge features (Messages)
        # Shape: [num_edges, hidden_size]
        h_interaction = self.edge_updater(pair_features)
        
        
        # --- STEP 2: Aggregation (Message Passing) ---
        
        # Track Aggregation: Sum up all interactions for each Track
        # "How many valid hits did I find? What is the total signal?"
        # src is the Track index
        aggr_to_track = scatter(h_interaction, src, dim=0, dim_size=h_track.size(0), reduce='add')
        
        # Hit Aggregation: Sum up all interactions for each Hit
        # "How many tracks claim me?" (Helps resolve ambiguity if tracks overlap)
        # dst is the Hit index
        aggr_to_hit = scatter(h_interaction, dst, dim=0, dim_size=h_hit.size(0), reduce='add')
        
        
        # --- STEP 3: State Updates ---
        
        # Update Tracks with the aggregated info
        h_track_updated = self.track_updater(torch.cat([h_track, aggr_to_track], dim=1))
        
        # Update Hits with the aggregated info
        h_hit_updated = self.hit_updater(torch.cat([h_hit, aggr_to_hit], dim=1))
        
        
        # --- STEP 4: Return Context for Final Classification ---
        # We re-construct the pair representation using the UPDATED node features.
        # This gives the final classifier the most refined information.
        final_edge_representation = torch.cat([h_track_updated[src], h_hit_updated[dst], h_edge], dim=1)
        
        return final_edge_representation

# Basic model wrapper for HeteroData
class GraphTORCHModel(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        # Track: [x, y, dx, dy, p, t0] -> 6
        # Hit: [x, y, t] -> 3
        # Edge: [dt, dx, dy] -> 3
        self.model = InteractionNetwork(6, 3, 3, hidden_size)
        
    def forward(self, data):
        # Unpack HeteroData
        x_track = data['track'].x
        x_hit = data['hit'].x
        edge_index = data['track', 'to', 'hit'].edge_index
        edge_attr = data['track', 'to', 'hit'].edge_attr
        
        return self.model(x_track, x_hit, edge_index, edge_attr)
