class ImplicitNetwork(nn.Module):
    def __init__(
            self,
            d_in,
            d_out,
            width,
            depth,
            geometric_init=True,
            bias=1.0,
            skip_in=-1,
            weight_norm=True,
            multires=0,
            pose_cond_layer=-1,
            pose_cond_dim=-1,
            pose_embed_dim=-1,
            shape_cond_layer=-1,
            shape_cond_dim=-1,
            shape_embed_dim=-1,
            latent_cond_layer=-1,
            latent_cond_dim=-1,
            latent_embed_dim=-1,
            feat_cond_dim=0,
            feat_cond_layer=[],
            dropout = 0,
            use_hash_encoding=True,
            **kwargs
    ):
        super().__init__()
        dims = [d_in] + [width]*depth + [d_out]
        self.embed_fn = None
        
        if use_hash_encoding:
            self.embed_fn = HashEmbedder(bounding_box_scal, num_levels, level_dim)
            input_ch = self.embed_fn.output_dim
            dims[0] = input_ch
        elif multires > 0:
            # 原有的 Positional Encoding 逻辑
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            dims[0] = input_ch
        # --- 修改结束 ---

        self.dropout = dropout
        
        # ... (以下代码保持不变，直到 self.num_layers 循环) ...
        self.cond_names = []

        self.pose_cond_layer = pose_cond_layer
        # ... (中间省略，保持原样) ...