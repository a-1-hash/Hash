import copy

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from skimage import measure

from hit.utils.extract_pretrained_gdna import get_state_dict
from hit.utils.metrics import occ2sdf
from hit.utils.slice_extractor import SliceLevelSet
from hit.utils.smpl_utils import get_skinning_weights, weights2colors, x_pose_like
from hit.utils.tensors import cond_create
from hit.utils.figures import tissue_palette

from hit.model.deformer import ForwardDeformer, skinning
from hit.model.helpers import expand_cond
from hit.model.network import ImplicitNetwork
from hit.model.generator import Generator
from hit.training.mri_sampling_utils import load_occupancy

#new test
from .surface_encoder import SurfaceFeatureEncoder, knn_interpolate


class HITModel(torch.nn.Module):

    def __init__(self, train_cfg, smpl) -> None:
        super().__init__()
        self.train_cfg = train_cfg
        self.apply_compression = True
        self.smpl = smpl

        ###################### Load the configuration for all the submodules MLP networks

        lbs_network_conf = train_cfg.networks.lbs
        beta_network_conf = train_cfg.networks.beta  # Beta displacement field config

        # Compressor config
        if self.train_cfg['compressor']:
            compressor_conf = train_cfg.networks.compression
        else:
            compressor_conf = None

        # Pose dep blend shapes config
        if self.train_cfg.pose_bs is True:
            pose_bs_conf = train_cfg.networks.pose_bs
        else:
            pose_bs_conf = None

        if self.train_cfg['forward_beta_mlp']:
            fwd_beta_conf = train_cfg.networks.beta_fwd
            self.fwd_beta = ImplicitNetwork(**fwd_beta_conf)
        else:
            self.fwd_beta = None

        # tissue occupancy prediction config
        tissue_cfg = train_cfg.networks.tissue
        tissue_cfg.update(
            {'d_out': len(self.train_cfg.mri_labels)})  # This network outputs a class score of each tissue

        #test
        # ==== 融合局部点云特征配置 ====
        self.use_surface_feats = train_cfg.get('use_surface_feats', True)
        from omegaconf import OmegaConf
        tissue_kwargs = OmegaConf.to_container(tissue_cfg, resolve=True)

        if self.use_surface_feats:
            self.surface_encoder = SurfaceFeatureEncoder(k=16, out_dim=128)
            tissue_kwargs.update({
                'feat_cond_dim': 128,
                'feat_cond_layer': [0]
            })
        else:
            self.surface_encoder = None
        # ==============================
        self.tissues_occ = ImplicitNetwork(**tissue_kwargs)



        if self.train_cfg['use_generator'] == False:
            self.generator = None
        else:
            self.generator = Generator(64)

        if train_cfg['mri_values']:
            # Instanciate an MLP that predicts the MRI values
            mri_val_net_conf = train_cfg.networks.mri_val
            self.mri_val_net = ImplicitNetwork(**mri_val_net_conf)
        else:
            self.mri_val_net = None

        self.deformer = ForwardDeformer(lbs_network_conf=lbs_network_conf,
                                        beta_network_conf=beta_network_conf,
                                        compressor_conf=compressor_conf,
                                        pose_bs_conf=pose_bs_conf,
                                        nb_root_init=train_cfg['root_init'],
                                        apply_compression=self.apply_compression)

    def initialize(self, pretrained=True, train_cfg=None, device=None, checkpoint_path=None):
        # self = COAPBodyModel(parametric_body, train_cfg)
        gender = self.smpl.gender

        if device is not None:
            self.smpl = self.smpl.to(device=device)

        if checkpoint_path is not None:
            print('Using checkpoint from ', checkpoint_path)
            checkpoint = checkpoint_path
            state_dict = torch.load(checkpoint)
            self.load_state_dict(state_dict['state_dict'])

        else:
            if train_cfg:
                print('Loading pretrained weights for the deformer...')
                if train_cfg['load_pretrained_lbs'] == True:
                    self.deformer.lbs_network.load_state_dict(get_state_dict(to_extract='deformer.lbs_network',
                                                                             source=f'pretrained_{gender}'))  # undo beta
                if train_cfg['load_pretrained_disp'] == True:
                    self.deformer.disp_network.load_state_dict(
                        get_state_dict(to_extract='deformer.disp_network', source=f'pretrained_{gender}'))  # undo beta
                if train_cfg['compressor'] and train_cfg['load_pretrained_compressor'] == True:
                    self.deformer.compressor.load_state_dict(get_state_dict(to_extract='deformer.compressor',
                                                                            source=f'pretrained_compressor_{gender}'))  # undo beta
                    self.deformer.compressor.load_state_dict(get_state_dict(to_extract='deformer.compressor',
                                                                            source=f'pretrained_compressor_{gender}'))  # undo beta

    def forward_rigged(self, betas, body_pose=None, global_orient=None, transl=None, do_compress=False,
                       mise_resolution0=32, mise_depth=3, **kwargs):

        # smpl shaped in xpose
        # smpl = MySmpl(model_path=cg.smplx_models_path, modelgender=self.smpl.gender, device=betas.device)
        smpl = self.smpl
        smpl_output_xpose = smpl.forward(betas=betas, body_pose=smpl.x_cano().to(betas.device), global_orient=None,
                                         transl=None)

        # smpl shaped and posed
        smpl_output = smpl.forward(betas=betas, body_pose=body_pose, global_orient=global_orient, transl=transl)
        tfs = smpl_output.tfs

        mesh_p_list = []
        mesh_c_list = []
        for ci, c_label in enumerate(self.train_cfg.mri_labels):
            if c_label != 'NO':
                mesh_s = self.extract_mesh(smpl_output_xpose, channel=ci, grid_res=64,
                                           use_mise=True, mise_resolution0=mise_resolution0,
                                           mise_depth=mise_depth, batch=None,
                                           template=False,
                                           unposed=True,  # The compression should be applied after posing
                                           color_mode='compression')[0]

                mesh_p = self.pose_unposed_tissue_mesh(mesh_s, smpl_output, do_compress=do_compress)

                # import ipdb; ipdb.set_trace()
                mesh_p_list.append(mesh_p)
                mesh_c_list.append(mesh_s)
        return mesh_p_list, mesh_c_list

    def extract_shaped_mesh(self, smpl_output, channel=1, grid_res=64, max_queries=None, use_mise=False,
                            mise_resolution0=32, bound_by_smpl=False):

        body_pose = self.smpl.forward_canonical(betas=torch.zeros_like(smpl_output.betas)).body_pose
        smpl_output_xpose = self.smpl.forward(betas=smpl_output.betas,
                                              body_pose=body_pose,
                                              global_orient=None, transl=None)

        mesh_s_list = self.extract_mesh(smpl_output_xpose, channel=channel, grid_res=64,
                                        use_mise=True, mise_resolution0=mise_resolution0,
                                        mise_depth=3, batch=None,
                                        template=False,
                                        unposed=True,  # The compression should be applied after posing
                                        color_mode='compression',
                                        bound_by_smpl=bound_by_smpl)
        return mesh_s_list

    def pose_unposed_tissue_mesh(self, mesh_s, smpl_output, do_compress=False):
        betas = smpl_output.betas
        body_pose = smpl_output.body_pose
        tfs = smpl_output.tfs
        cond = cond_create(betas, body_pose, self.generator, self.smpl)
        x_s = torch.FloatTensor(np.array(mesh_s.vertices)).to(betas.device).unsqueeze(0)
        x_c = self.deformer.disp_network(x_s, cond) + x_s

        # skinning
        w = self.deformer.query_weights(x_c, {'latent': cond['lbs'], 'betas': cond['betas'] * 0})
        xd = skinning(x_s[0], w[0], tfs, inverse=False)

        if do_compress:
            raise DeprecationWarning("This is wrong, do not use. Instead use the posed extraction")
            d_p = self.deformer.compressor(x_s, cond)
            xd = xd + d_p[0]

        mesh_p = trimesh.Trimesh(vertices=xd.detach().cpu().numpy(), faces=mesh_s.faces)
        return mesh_p

    def can_occ_query(self, pts_c, n_batch, n_dim, cond, both=False, surface_feats=None):
        """
        已修复：确保特征提取在 Canonical 空间进行，并适配 tissues_occ 接口
        """
        input_feat = None

        # 逻辑：只有开启了特征融合功能且确实有 encoder 时才执行
        if getattr(self, 'use_surface_feats', False) and self.surface_encoder is not None:
            if surface_feats is None:
                # 核心修复：如果外部没传，内部提取时必须强制使用 Canonical 顶点
                # 这样才能保证查询点 pts_c 与体表顶点 surface_verts 空间对齐
                cano_smpl = self.smpl.forward_canonical(betas=cond['betas'])
                surface_verts = cano_smpl.vertices
                # 提取体表高频特征
                raw_surface_feats = self.surface_encoder(surface_verts)
                # 执行插值 (使用带 Chunking 和 Distance Mask 的版本)
                pts_c_flat = pts_c.reshape(n_batch, -1, n_dim)
                input_feat = knn_interpolate(pts_c_flat, surface_verts, raw_surface_feats, k=16, chunk_size=10000)
            else:
                # 外部已经传了插值好的特征（通常是 query 函数传进来的）
                input_feat = surface_feats

        # 辅助函数：统一调用 MLP 接口
        def call_net(net, p, f):
            return net(
                p.reshape((n_batch, -1, n_dim)),
                cond={'latent': cond['latent']},
                input_feat=f,
                mask=None,
                val_pad=-1000,
                return_feat=False,
                spatial_feat=True,
                normalize=True
            )

        if both:
            if (self.train_cfg['to_train'] in ['pretrain', 'compression']):
                occ_smpl_pd = call_net(self.tissues_occ, pts_c, input_feat)
                occ_tissue_pd = occ_smpl_pd
            else:
                occ_tissue_pd = call_net(self.tissues_occ, pts_c, input_feat)
                occ_smpl_pd = None
            return occ_smpl_pd, occ_tissue_pd

        if self.train_cfg['to_train'] == 'pretrain':
            occ_pd = call_net(self.network, pts_c, input_feat)
        else:
            occ_pd = call_net(self.tissues_occ, pts_c, input_feat)

        return occ_pd

    def canonicalize_from_similar(self, x_p, xx_c, smpl_tfs, cond, undo_shape=True):
        """
        xf_p : Points to unpose
        xv_p : Posed points to get w from
        smpl_tfs : Bone transforms of the posed smpl
        undo_shape : if true, undo beta
        return :  unposed points x_c


        """

        B = x_p.shape[0]
        T = x_p.shape[1]

        w = self.deformer.query_weights(xx_c, {'latent': cond['lbs'], 'betas': cond['betas']})

        x_p_flat = x_p.reshape(-1, 3)
        tfs = expand_cond(smpl_tfs, x_p)
        tfs_flat = tfs.reshape(-1, tfs.shape[-3], 4, 4)
        w_flat = w.reshape(-1, w.shape[-1])

        # import ipdb; ipdb.set_trace()
        x_s_flat = skinning(x_p_flat, w_flat, tfs_flat, inverse=True)
        x_s = x_s_flat.reshape(B, T, 3)

        if undo_shape:
            x_c = self.deformer.disp_network(x_s, cond) + x_s
            return x_c
        else:
            return x_s

    def query(self, points, smpl_output, part_id=None, skinning_weights=None, eval_mode=False, template=False,
              unposed=False, apply_compression=True):
        """
        已修复：强制在 Canonical 空间提取特征，并实现平滑性扰动闭环
        """
        T = points.shape[1]
        n_dim = 3
        B = points.shape[0]
        n_batch = B

        pts_d = points
        mask = None
        cond = cond_create(smpl_output.betas, smpl_output.body_pose, None, self.smpl)
        smpl_tfs = smpl_output.tfs

        if mask is None:
            mask = torch.ones((n_batch, T), device=pts_d.device, dtype=torch.bool)

        # 1. 确定 Canonical 空间坐标 pts_c (这部分保留原逻辑)
        if template:
            pts_c = pts_d
        elif unposed:
            pts_c = self.deformer.query_cano(points,
                                             {'betas': cond['betas'],
                                              'thetas': cond['thetas']},
                                             mask=None)
        else:
            assert not skinning_weights is None
            pts_c, others = self.deformer.forward(pts_d,
                                                  {'betas': cond['betas'],
                                                   'thetas': cond['thetas'],
                                                   'latent': cond['lbs']},
                                                  tfs=smpl_tfs,
                                                  mask=mask,
                                                  eval_mode=eval_mode,
                                                  part_id=part_id,
                                                  skinning_weights=skinning_weights)

            if pts_c.shape[2] == 1:
                pts_c = pts_c[:, :, 0]
            else:
                if not (self.deformer.compressor is None or not self.apply_compression or not apply_compression or
                        self.train_cfg['skip_compression']):
                    xf_c = pts_c[:, :, 0]
                    xf_p = pts_d
                    xf_s = self.canonicalize_from_similar(xf_p, xf_c, smpl_tfs, cond, undo_shape=False)
                    if self.train_cfg['no_comp_grad']:
                        with torch.no_grad():
                            d_p = self.deformer.compressor(xf_s, cond)
                    else:
                        d_p = self.deformer.compressor(xf_s, cond)
                    xv_p = xf_p + d_p
                    pts_c = self.canonicalize_from_similar(xv_p, xf_c, smpl_tfs, cond)
                else:
                    pts_c = pts_c[:, :, 0]

        # 2. 执行组织预测 (核心修复点)
        input_feat = None
        if getattr(self, 'use_surface_feats', False) and self.surface_encoder is not None:
            # 关键：生成标准 T-pose/A-pose 顶点，确保空间绝对对齐
            cano_smpl = self.smpl.forward_canonical(betas=cond['betas'])
            surface_pts = cano_smpl.vertices
            # 提取标准表面特征
            surface_feats_raw = self.surface_encoder(surface_pts)

            # 在 Canonical 空间进行插值，解决 4.png 的错位问题
            pts_c_flat = pts_c.reshape(n_batch, -1, n_dim)
            input_feat = knn_interpolate(pts_c_flat, surface_pts, surface_feats_raw, k=16, chunk_size=10000)

        # 正常查询预测
        occ_smpl_pd, occ_pd = self.can_occ_query(pts_c, n_batch, n_dim, cond, both=True, surface_feats=input_feat)

        # 4. 封装输出
        weights = self.deformer.query_weights(pts_c, {'latent': cond['lbs'], 'betas': cond['betas']})
        output = {
            'pred_occ': occ_pd,
            'pts_c': pts_c,
            'smpl_occ': occ_smpl_pd,
            'weights': weights
        }

        return output

    @staticmethod
    def batchify_smpl_output(smpl_output):
        b_smpl_output_list = []
        batch_size = smpl_output.vertices.shape[0]
        for b_ind in range(batch_size):
            b_smpl_output_list.append(copy.copy(smpl_output))
            for key in b_smpl_output_list[
                -1].__dict__.keys():  # callling __dict__.keys() is necessary to support v_free

                # print(key)
                val = getattr(smpl_output, key)
                if torch.is_tensor(val):
                    val = val[b_ind:b_ind + 1].clone()
                setattr(b_smpl_output_list[-1], key, val)
        return b_smpl_output_list

    @torch.no_grad()
    def pts_distances(self, smpl_output, points):

        if isinstance(smpl_output.vertices, torch.Tensor):
            verts_numpy = smpl_output.vertices.squeeze().cpu().numpy()
        else:

            verts_numpy = smpl_output.vertices
        mesh = trimesh.Trimesh(verts_numpy, self.smpl.faces, process=False)
        from leap.tools.libmesh import check_mesh_contains
        is_inside = check_mesh_contains(mesh, points).astype(float)

        # Compute distances from points to mesh surface using trimesh
        if isinstance(points, torch.Tensor):
            points_numpy = points.squeeze().cpu().numpy()
        else:
            points_numpy = points
        proximity = trimesh.proximity.ProximityQuery(mesh)
        signed_dist = proximity.signed_distance(points_numpy)

        is_inside = torch.FloatTensor(is_inside).to(smpl_output.vertices.device)
        signed_dist = torch.FloatTensor(signed_dist).to(smpl_output.vertices.device)

        return is_inside, signed_dist

    def query_gt(self, query_points, batch, b_ind=None):

        gt_occ = load_occupancy(batch, query_points.cpu().numpy(), interp_order=0, b_ind=b_ind)[
            0]  # because load_occupancy returns [occ, body_mask]
        return torch.FloatTensor(gt_occ).to(query_points.device)

    @torch.no_grad()
    def extract_mesh(self, smpl_output, channel=1, grid_res=64, max_queries=None, use_mise=True, mise_resolution0=32,
                     mise_depth=3, batch=None, template=False, unposed=False, color_mode='lbs', bound_by_smpl=False):

        if max_queries is None:
            max_queries = int(self.train_cfg['max_queries'])
        scale = 1.1  # padding

        act = lambda x: torch.log(x / (1 - x + 1e-6) + 1e-6)  # revert sigmoid
        level = 0.5
        occ_list = []

        verts = smpl_output.vertices
        B = verts.shape[0]
        device = verts.device
        b_smpl_output_list = self.batchify_smpl_output(smpl_output)

        if template == True:
            # All the meshes are the same, only extract the first one
            B = 1
            b_smpl_output_list = [b_smpl_output_list[1]]

        b_min, b_max = verts.min(dim=1).values, verts.max(dim=1).values  # B,3
        gt_center = ((b_min + b_max) * 0.5).cpu()  # B,3
        gt_scale = (b_max - b_min).max(dim=-1, keepdim=True).values.cpu()  # (B,1)
        gt_scale_gpu, gt_center_gpu = gt_scale.to(device), gt_center.to(device)

        mesh = trimesh.Trimesh(smpl_output.vertices.detach().cpu().squeeze().numpy(), self.smpl.faces, process=False)

        if use_mise:
            from leap.tools.libmise import MISE
            value_grid_list = []
            for b_ind in range(B):
                mesh_extractor = MISE(mise_resolution0, mise_depth, level)
                points = mesh_extractor.query()
                print('Running marching cube...')
                while points.shape[0] != 0:
                    grid3d = torch.FloatTensor(points).to(device)
                    grid3d = scale * (grid3d / mesh_extractor.resolution - 0.5).reshape(1, -1, 3)  # [-0.5, 0.5]*scale
                    grid3d = grid3d * gt_scale_gpu[b_ind].reshape(1, 1, 1) + gt_center_gpu[b_ind].reshape(1, 1, 3)

                    # check occupancy for sampled points
                    occ_hats = []
                    for pts in torch.split(grid3d, max_queries, dim=1):

                        # import ipdb; ipdb.set_trace()
                        skinning_weights, part_id = get_skinning_weights(pts[0].cpu().numpy(),
                                                                         b_smpl_output_list[b_ind].vertices[
                                                                             0].cpu().numpy(),
                                                                         self.smpl)
                        skinning_weights = torch.FloatTensor(skinning_weights).to(device)

                        if self.train_cfg.use_precomputed_dist:
                            is_inside, signed_dist = self.pts_distances(b_smpl_output_list[b_ind],
                                                                        pts.squeeze(0).cpu().numpy())
                        else:
                            is_inside, signed_dist = None, None

                        if batch is not None:
                            # Fetch GT occupancy
                            pred = self.query_gt(pts.to(device=device), batch, b_ind)
                            occ_tissue = (pred == channel).float()
                            occ_hats.append(occ_tissue.cpu())

                        else:
                            # import ipdb; ipdb.set_trace()
                            output = self.query(pts.to(device=device), b_smpl_output_list[b_ind], eval_mode=True,
                                                template=template, unposed=unposed,
                                                part_id=part_id, skinning_weights=skinning_weights)
                            if channel == -1:
                                # import ipdb; ipdb.set_trace()
                                pred = output['smpl_occ'].squeeze(-1)
                            else:
                                pred = output['pred_occ']

                            if len(self.train_cfg.mri_labels) > 1 and channel != -1:
                                sdf = occ2sdf(pred, channel)

                            else:
                                sdf = torch.sigmoid((pred.cpu().squeeze(0)))

                            occ_hats.append(sdf)
                    # print('Done.')
                    values = torch.cat(occ_hats, dim=0).numpy().astype(np.float64)
                    # sample points again
                    mesh_extractor.update(points, values)
                    points = mesh_extractor.query()

                value_grid_list.append(mesh_extractor.to_dense())
            value_grid = np.stack(value_grid_list)
            grid_res = mesh_extractor.resolution
        else:
            raise NotImplementedError('Only MISE is supported for mesh extraction.')

        # extract meshes
        mesh_list = []

        exception_error = ''
        for b_ind in range(B):
            try:
                verts, faces, normals, values = measure.marching_cubes(volume=value_grid[b_ind],
                                                                       gradient_direction='ascent', level=level)
            except Exception as e:
                exception_error = e
                print(f'Marching cubes failed for one of the batch item, no mesh will be returned: {exception_error}')
                return []

            # vertices to world space
            verts = scale * (verts / (grid_res - 1) - 0.5)
            verts = verts * gt_scale[b_ind].item() + gt_center[b_ind].cpu().numpy()

            verts_batched = torch.tensor(verts).to(smpl_output.betas.device).unsqueeze(0)
            if template == True:
                # color verts
                cond = cond_create(betas=smpl_output.betas[0].unsqueeze(0))
                if color_mode == 'compression' and self.train_cfg['compressor']:

                    # Marching cube conmpression
                    comp = self.deformer.compressor(verts_batched, {'betas': cond['betas'] * 0})[0].cpu().numpy()

                    # # Template compression
                    # comp = self.deformer.compressor(smpl_output.vertices, {'betas': cond['betas']*0})[0].cpu().numpy()
                    # cols = np.abs(comp) / np.abs(comp).max()
                    # mesh = trimesh.Trimesh(smpl_output.vertices[0].cpu().numpy(), self.smpl.faces, vertex_colors=cols, process=False)
                    # mesh.export('/tmp/comp_mesh_template.ply')

                    # cols = comp / torch.linalg.norm(comp, axis=-1).unsqueeze(-1)
                    cols = np.abs(comp) / np.abs(comp).max()
                    mesh = trimesh.Trimesh(verts, faces, vertex_colors=cols, process=False)
                    mesh.export('/tmp/comp_mesh_mc.ply')


                else:
                    w_pd = self.deformer.query_weights(verts_batched,
                                                       {'latent': cond['lbs'], 'betas': cond['betas'] * 0})
                    cols = weights2colors(w_pd[0].cpu().numpy())
                    mesh = trimesh.Trimesh(verts, faces, vertex_colors=cols, process=False)
                    mesh.export('/tmp/lbs_mesh.ply')


            else:
                # w_pd = self.deformer.query_weights(verts_batched, {'latent': cond['lbs'], 'betas': cond['betas']*0})
                # cols = weights2colors(w_pd[0].cpu().numpy())
                cols = tissue_palette[channel]
                mesh = trimesh.Trimesh(verts, faces, vertex_colors=cols, process=False)
            # color meshes
            # vertex_colors = self.color_points(torch.from_numpy(verts).reshape(1, -1, 3).to(device), b_smpl_output_list[b_ind], max_queries)[0]
            mesh_list.append(mesh)
        return mesh_list

    @torch.no_grad()
    def generate_canonical_mesh(self, batch, channel_index):

        smpl_data = {}
        smpl_data['betas'] = torch.zeros_like(batch['betas'])
        smpl_data['body_pose'] = x_pose_like(batch['body_pose'])
        smpl_data['global_orient'] = torch.zeros_like(batch['global_orient'])
        smpl_data['transl'] = torch.zeros_like(batch['transl'])
        smpl_data['global_orient_init'] = torch.zeros_like(batch['global_orient_init'])
        smpl_output = self.smpl(**smpl_data, return_verts=True, return_full_pose=True)

        meshes = self.extract_mesh(smpl_output, channel=channel_index, use_mise=True, template=True)
        if len(meshes) > 0:
            return meshes[0]
        else:
            # The mesh extraction failed, return None
            return None

    @torch.no_grad()
    def evaluate_slice(self, batch, smpl_output, z0, axis='z', values=["occ", "sw", "beta", "fwd_beta"], res=0.01):
        """
        Infer the different values on a slice of the 3D space
        Args:
            batch : dict containing the input smpl parameters
            smpl_output : the output of the SMPL model
            z0 : the z coordinate of the slice
            axis : the axis of the slice
            values : the values to infer on the slice, can contain :
                "occ" (occupancy)
                "sw" (skinning weights)
            res : the size of a slice pixel (in meters)
        Returns:
            out_images : list of pillow images of the different values
        """

        sl = SliceLevelSet(nbins=10, xbounds=[-0.2, 0.2], ybounds=[-0.2, 0.2], z_bounds=[-0.2, 0.2], res=res)

        xc = sl.gen_slice_points(z0=z0, axis=axis)
        xc_batch = torch.FloatTensor(xc).to(batch['betas'].device).unsqueeze(0).expand(batch['betas'].shape[0], -1, -1)
        cond = cond_create(batch['betas'], batch['body_pose'], self.generator)

        occ = None
        if "occ" in values:
            # Query HIT
            output = self.query(xc_batch, smpl_output, unposed=True, eval_mode=True)

            # Gather all values to monitor
            pts_c = output['pts_c']
            sw = output['weights']
            occ = output['pred_occ']
            # disp = self.deformer.disp_network(xc_batch, cond)
            if 'beta' in values:
                disp = self.deformer.disp_network(xc_batch, cond)
            if "comp" in values:
                comp = self.deformer.compressor(xc_batch, cond)
        else:

            if "sw" in values:
                sw = self.deformer.query_weights(xc_batch, cond={})
            if "beta" in values:
                disp = self.deformer.query_betadisp(xc_batch, cond={'betas': cond['betas']})

        if "fwd_beta" in values:
            B_flat = self.fwd_beta(xc_batch, cond)
            B_xc = B_flat.view(B_flat.shape[0], xc_batch.shape[1], 10, 3)
            beta_rep = batch['betas'].repeat(xc_batch.shape[1], 1).reshape(-1, 1, 10)
            B_xc_concat = B_xc.reshape(-1, 10, 3)  # B*N, 10, 3
            fwd_disp_flat = torch.bmm(beta_rep, B_xc_concat)  # B*N, 1, 3
            fwd_disp = fwd_disp_flat.view(-1, xc_batch.shape[1], 3)  # B, N, 3

        # import ipdb; ipdb.set_trace()
        out_images = []
        for value in values:
            if value == "occ":
                image = sl.plot_occupancy(occ)
                out_images.append(image)

            elif value == 'sw':
                image = sl.plot_skinning_weights(sw, occ)
                out_images.append(image)

            elif value == 'beta':
                image = sl.plot_disp_field(disp, occ, quiver=True)
                out_images.append(image)

            elif value == 'fwd_beta':
                image = sl.plot_disp_field(fwd_disp, occ, quiver=True)
                out_images.append(image)

            elif value == 'comp':
                image = sl.plot_disp_field(comp, occ, quiver=True, is_compression=True)
                out_images.append(image)

            else:
                print(f"Value {value} not recognized")

        return out_images


