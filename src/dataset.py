
import uproot
import torch
import numpy as np
import awkward as ak
from torch_geometric.data import HeteroData, Dataset
import os

class TorchDataset(Dataset):
    def __init__(self, root, filename=None, transform=None, pre_transform=None):
        self.filename = filename
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return [self.filename] if self.filename else []

    @property
    def processed_file_names(self):
        if self.filename:
            return [f'data_{i}.pt' for i in range(1)] # Simplified for now, usually one per event or chunk
        return ['mock_data.pt']

    def process(self):
        # Constants
        m_pion = 139.57018 # MeV/c^2
        c_speed = 299.792458 # mm/ns (vacuum speed of light)
        
        if self.filename is None:
            # Generate dummy data for verification
            print("Generating dummy data...")
            num_events = 100
            for i in range(num_events):
                data = self._generate_dummy_event(m_pion, c_speed)
                torch.save(data, os.path.join(self.processed_dir, f'data_{i}.pt'))
            return

        # Real data loading
        print(f"Loading data from {self.filename}...")
        with uproot.open(self.filename) as file:
            # Assumes a tree name, e.g., 'events' or similar. 
            # We need to list keys to be sure, but for now assuming the first tree found or 'tree'
            tree = file[file.keys()[0]] 
            
            # Read arrays
            # Expecting arrays of arrays (jagged arrays) because mult features per event
            # User said: "each entry is a collection of track information and hit information"
            
            # Using specific column names provided by user
            track_branches = ["xCoor", "yCoor", "xDir", "yDir", "momentum", "t0", "pathlength", "trackID"]
            hit_branches = ["hitX", "hitY", "hitT", "hitTrackId"]
            
            # Load in chunks to manage memory if large file
            for i, batch in enumerate(tree.iterate(track_branches + hit_branches, library="ak")):
                # Iterate over events in batch
                for j in range(len(batch)):
                    event = batch[j]
                    
                    # Extract Track Data
                    # Shape: (NumTracks, 7) + 1 ID
                    tracks_np = np.stack([event[b] for b in track_branches], axis=1)
                    
                    # Extract Hit Data
                    # Shape: (NumHits, 3) + 1 ID
                    hits_np = np.stack([event[b] for b in hit_branches], axis=1)
                    
                    data = self._build_event(tracks_np, hits_np, m_pion, c_speed)
                    torch.save(data, os.path.join(self.processed_dir, f'data_{i*len(batch) + j}.pt'))


    def _build_event(self, tracks_np, hits_np, m_pion, c_speed):
        # Convert to tensor
        tracks = torch.tensor(tracks_np, dtype=torch.float) # [N_t, 8] including ID
        hits = torch.tensor(hits_np, dtype=torch.float)     # [N_h, 4] including ID
        
        num_tracks = tracks.shape[0]
        num_hits = hits.shape[0]
        
        # Node Features
        # Track: [x, y, dx, dy, p, t0] (Exclude pathlength and ID from features)
        x_track = tracks[:, :6] 
        
        # Hit: [x, y, t] (Exclude ID)
        x_hit = hits[:, :3]
        
        # Edge Construction (Fully Connected Bipartite)
        # Source: Tracks (indices 0..N_t-1)
        # Target: Hits (indices 0..N_h-1)
        
        # Create meshgrid of indices
        track_indices = torch.arange(num_tracks, dtype=torch.long).repeat_interleave(num_hits)
        hit_indices = torch.arange(num_hits, dtype=torch.long).repeat(num_tracks)
        
        edge_index = torch.stack([track_indices, hit_indices], dim=0)
        
        # Calculate Delta T
        pathlength = tracks[track_indices, 6]
        t0 = tracks[track_indices, 5]
        momentum = tracks[track_indices, 4]
        
        hit_t = hits[hit_indices, 2]
        
        # ToF Calculation
        # tof = (L / c) * sqrt(1 + (m / p)^2)
        inv_beta_sq = 1 + (m_pion / (momentum + 1e-6)) ** 2
        tof = (pathlength / c_speed) * torch.sqrt(inv_beta_sq)
        
        delta_t = hit_t - (t0 + tof)
        
        edge_attr = delta_t.view(-1, 1) # [E, 1]
        
        # Ground Truth Labels
        # Check if trackID matches hitTrackId for each edge
        track_ids = tracks[track_indices, 7] # ID is at index 7
        hit_track_ids = hits[hit_indices, 3] # ID is at index 3
        
        # Label is 1 if IDs match, else 0
        edge_label = (track_ids == hit_track_ids).float().view(-1)
        
        data = HeteroData()
        data['track'].x = x_track
        data['hit'].x = x_hit
        data['track', 'to', 'hit'].edge_index = edge_index
        data['track', 'to', 'hit'].edge_attr = edge_attr
        data['track', 'to', 'hit'].y = edge_label # Add labels
        
        return data

    def _generate_dummy_event(self, m_pion, c_speed):
        num_tracks = np.random.randint(2, 10)
        num_hits = np.random.randint(20, 50)
        
        # Tracks: [x, y, dx, dy, p, t0, L, trackID]
        tracks_np = np.random.rand(num_tracks, 8).astype(np.float32)
        tracks_np[:, 4] *= 10000 # Momentum
        tracks_np[:, 5] *= 10 # t0
        tracks_np[:, 6] *= 1000 # L
        tracks_np[:, 7] = np.arange(num_tracks) # unique IDs
        
        # Hits: [x, y, t, hitTrackId]
        hits_np = np.random.rand(num_hits, 4).astype(np.float32)
        hits_np[:, 2] *= 50 # t
        
        # Assign random track IDs to hits to create matches
        # Some hits might be noise (ID = -1), but let's assume they all come from tracks for now
        hits_np[:, 3] = np.random.choice(tracks_np[:, 7], size=num_hits)
        
        # Make physics somewhat consistent for true matches?
        # For dummy data, we just want code to run, so physics consistency isn't strictly required for the pipeline test.
        # But let's leave it random.
        
        return self._build_event(tracks_np, hits_np, m_pion, c_speed)

    def len(self):
        # Return number of processed files
        return len([f for f in os.listdir(self.processed_dir) if f.startswith('data_')])

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'), weights_only=False)
        return data

if __name__ == "__main__":
    # Test
    dataset = TorchDataset(root='data')
    dataset.process() # Force process for test
    print(f"Dataset length: {len(dataset)}")
    print(f"Sample data: {dataset[0]}")
