
import uproot
import torch
import numpy as np
from torch_geometric.data import HeteroData, InMemoryDataset

class TorchDataset(InMemoryDataset):
    def __init__(self, root, filename=None, treename=None, transform=None, pre_transform=None):
        self.filename = filename
        self.treenames = treename
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return [self.filename] if self.filename else []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        # Constants
        m_pion = 139.57018 # MeV/c^2
        c_speed = 299.792458 # mm/ns (vacuum speed of light)
        
        data_list = []
        
        with uproot.open(self.filename) as file:
            tree = file[self.treenames] 
            
            # Read arrays
            track_branches = ["xCoor", "yCoor", "xDir", "yDir", "momentum", "t0", "pathlength", "trackID"]
            hit_branches = ["hitX", "hitY", "hitT", "hitTrackId"]
            
            # Load in chunks
            for i, batch in enumerate(tree.iterate(track_branches + hit_branches, library="ak")):
                print(f"Processing chunk {i}")
                for j in range(len(batch)):
                    event = batch[j]
                    
                    # Extract Data
                    tracks_np = np.stack([event[b] for b in track_branches], axis=1)
                    hits_np = np.stack([event[b] for b in hit_branches], axis=1)
                    
                    data = self._build_event(tracks_np, hits_np, m_pion, c_speed)
                    data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


    def _build_event(self, tracks_np, hits_np, m_pion, c_speed):
        tracks = torch.tensor(tracks_np, dtype=torch.float)
        hits = torch.tensor(hits_np, dtype=torch.float)
        
        # Ensure 2D shape [Num, Features] even if 0 or 1 item
        if tracks.dim() == 1 and tracks.numel() > 0:
             # If it was a single track but flattened, finding out correct width is tricky if 0.
             # But usually tracks_np comes from np.stack checks.
             # Let's rely on reshaping if we know feature count.
             # Track features: 8 (from track_branches)
             tracks = tracks.view(-1, 8)
        elif tracks.dim() == 1 and tracks.numel() == 0:
             tracks = tracks.view(0, 8)

        if hits.dim() == 1 and hits.numel() > 0:
             # Hit features: 4 (from hit_branches)
             hits = hits.view(-1, 4)
        elif hits.dim() == 1 and hits.numel() == 0:
             hits = hits.view(0, 4)
        
        num_tracks = tracks.shape[0]
        num_hits = hits.shape[0]
        
        # --- PREPROCESSING ---
        
        # Feature Indices:
        # Track: 0:x, 1:y, 2:dx, 3:dy, 4:p (GeV)
        # Hit:   0:x, 1:y, 2:t
        
        # 1. Scaling (Hardcoded ranges)
        # Goal: Map inputs to approx [-1, 1] range
        
        # Track Features
        x_track = tracks[:, :6].clone()
        x_track[:, 0] /= 330.0   # x (mm) [-330, 330] -> [-1, 1]
        x_track[:, 1] /= 1250.0  # y (mm) [-1250, 1250] -> [-1, 1]
        # dx, dy (indices 2,3) are unit vector components -> Already [-1, 1]
        x_track[:, 4] /= 100.0   # p (GeV) [0, 100] -> [0, 1]
        
        # Hit Features
        x_hit = hits[:, :3].clone()
        x_hit[:, 0] /= 330.0     # x (mm) [-330, 330] -> [-1, 1]
        x_hit[:, 1] /= 30.0      # y (mm) [-30, 30] -> [-1, 1] (Small PMT dimension?)
        x_hit[:, 2] /= 50.0      # t (ns) Scale to order 1
        
        # Edge Construction (Fully Connected Bipartite)
        track_indices = torch.arange(num_tracks, dtype=torch.long).repeat_interleave(num_hits)
        hit_indices = torch.arange(num_hits, dtype=torch.long).repeat(num_tracks)
        edge_index = torch.stack([track_indices, hit_indices], dim=0)
        
        # Delta T Calculation (Physics Logic)
        # Note: Use UN-SCALED values for physics calc!
        
        # Constants handling
        # m_pion is MeV. Momentum is GeV.
        # Convert m_pion to GeV to match momentum.
        m_pion_gev = m_pion / 1000.0
        
        pathlength = tracks[track_indices, 6] # Raw (mm)
        t0 = tracks[track_indices, 5]         # Raw t0 (ns)
        momentum = tracks[track_indices, 4]   # Raw p (GeV)
        
        hit_t = hits[hit_indices, 2]          # Raw hit time (ns)
        
        # ToF = L / (c * beta)
        # beta = p / E = p / sqrt(p^2 + m^2)
        beta = momentum / torch.sqrt(momentum**2 + m_pion_gev**2)
        tof = pathlength / (c_speed * beta)

        delta_t = hit_t - (t0 + tof)
        
        # Edge Feature Scaling
        # Let's scale by 20
        edge_attr = (delta_t / 20.0).view(-1, 1)
        
        # Labels
        track_ids = tracks[track_indices, 7]
        hit_track_ids = hits[hit_indices, 3]
        edge_label = (track_ids == hit_track_ids).float().view(-1)
        
        data = HeteroData()
        data['track'].x = x_track
        data['hit'].x = x_hit
        data['track', 'to', 'hit'].edge_index = edge_index
        data['track', 'to', 'hit'].edge_attr = edge_attr
        data['track', 'to', 'hit'].y = edge_label
        
        return data


if __name__ == "__main__":
    # Test
    # Note: This requires an actual file to work properly
    print("Please run this via train.py or ensure a root file exists.")
