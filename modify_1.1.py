
class HashEmbedder(nn.Module):
    def __init__(self, bounding_box_scale, num_levels, level_dim, base_resolution, log2_hashmap_size, max_resolution, primes):
        super().__init__()
        self.bounding_box_scale = bounding_box_scale
        self.num_levels = num_levels
        self.level_dim = level_dim
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.output_dim = num_levels * level_dim
        
        self.embeddings = nn.ModuleList([
            nn.Embedding(2**self.log2_hashmap_size, self.level_dim) 
            for _ in range(num_levels)
        ])
        

        self.b = np.exp((np.log(max_resolution) - np.log(base_resolution)) / (num_levels - 1))

        for i in range(num_levels):
            nn.init.uniform_(self.embeddings[i].weight, a=-1e-4, b=1e-4)

    def forward(self, x):

        x_norm = (x + self.bounding_box_scale) / (2 * self.bounding_box_scale)
        x_norm = torch.clamp(x_norm, 0.0, 1.0) 

        features = []
        primes = []

        for i in range(self.num_levels):
            resolution = int(np.floor(self.base_resolution * (self.b ** i)))
            

            scaled_x = x_norm * resolution
            

            x0 = torch.floor(scaled_x).long()

            x1 = x0 + 1
            
            weights = scaled_x - x0.float() 
            
            offsets = torch.tensor([[0,0,0], [0,0,1], [0,1,0], [0,1,1], 
                                    [1,0,0], [1,0,1], [1,1,0], [1,1,1]], device=x.device)
            
            current_level_feature = 0
            
            for j in range(8):
                corner_offset = offsets[j]
                corner_coords = x0 + corner_offset
                
                xor_result = (corner_coords[:, 0] * primes[0]) ^ \
                             (corner_coords[:, 1] * primes[1]) ^ \
                             (corner_coords[:, 2] * primes[2])
                hash_indices = xor_result % (2 ** self.log2_hashmap_size)
                

                corner_embed = self.embeddings[i](hash_indices) # [Batch, level_dim]
                
                w = torch.ones(x.shape[0], 1, device=x.device)
                for k in range(3): # x, y, z dimensions
                    if corner_offset[k] == 0:
                        w = w * (1 - weights[:, k:k+1])
                    else:
                        w = w * weights[:, k:k+1]
                
                current_level_feature += w * corner_embed
            
            features.append(current_level_feature)
            
        return torch.cat(features, dim=-1)

# -----------------------------------------------