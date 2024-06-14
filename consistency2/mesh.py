from pytorch3d.io import load_obj
import torch 
import numpy as np
import nvdiffrast.torch as dr
import xatlas

def transform_pos_batched(mtx, pos): # [b, 4, 4] [b, V, 3]
    t_mtx = torch.from_numpy(mtx).cuda() if isinstance(mtx, np.ndarray) else mtx
    posw = torch.cat([pos, torch.ones([pos.shape[0], pos.shape[1], 1]).cuda()], axis=2)    
    res = torch.matmul(posw, t_mtx.transpose(1,2)).to(torch.float32)
    return res

def normalize_depth(depth_map, min_val = 0.0):
    object_mask = depth_map != 0

    depth_map[object_mask] = ((1 - min_val) * (depth_map[object_mask] - depth_map[object_mask].min()) / (
            depth_map[object_mask].max() - depth_map[object_mask].min())) + min_val


    return depth_map

def position_verts(verts, trans_mat, shape_scale=1.2):
    verts = torch.matmul(verts, trans_mat)

    verts = verts - (verts.max(0).values + verts.min(0).values) * 0.5

    verts = verts / torch.sqrt(torch.sum(verts * verts, axis=1)).max()
    verts = verts * shape_scale

    return verts



class RenderSettings:
    def __init__(self,
                 resolution = 64,
                 interp_type = "linear", #["nearest", "linear", "mip"]
                 enable_mip = False,
                 max_mip_level = 10,
                 mip_level_bias = None,
                 absolute_bias_level = False,
                 latent = False,
                 channels = None,
                ):
        self.resolution = resolution
        self.enable_mip = enable_mip
        self.max_mip_level = max_mip_level
        self.mip_level_bias = mip_level_bias
        self.absolute_bias_level = absolute_bias_level
        self.latent = latent
        self.interp_type =  interp_type
        self.channels = channels
        if self.channels is None:
            if self.latent: self.channels = 4
            else: self.channels = 3
        
class Mesh:
    def __init__(self, mesh_config = None, device = "cuda"):


        trans_mat = mesh_config.get("trans_mat", torch.eye(3))
        mesh_path = mesh_config.get("obj", "")
        generate_uvs = mesh_config.get("generate_uvs", False)
        scale = mesh_config.get("scale", 1.2)
        verts, faces, aux = load_obj(mesh_path, device = device)
        
        trans_mat = trans_mat.to(verts.device)
        verts =  position_verts(verts, trans_mat, shape_scale=scale)
        tex_maps = aux.texture_images
        if tex_maps is not None and len(list(tex_maps.values())) > 0:
            texture_image = list(tex_maps.values())[0]
            texture_image = texture_image[None, ...].to(device)  # (1, H, W, 3)
        else:
            texture_image = 0.5 * torch.ones((1, 64,64,3), device = device, dtype = torch.float32)


        self.scale = scale
        self.tex = texture_image.contiguous()[0] # texture img
        self.pos_idx = faces.verts_idx.to(device).to(torch.int32).contiguous() # faces [F, 3]
        self.vtx_pos = verts.to(device).contiguous() # vertices [V, 3]
        self.device = device
        if (not generate_uvs) and (aux.verts_uvs is not None) and (faces.textures_idx is not None):
            print("use given tex coords")
            vtx_uv = aux.verts_uvs.to(device).contiguous()
            vtx_uv[:,1] = 1 - vtx_uv[:,1] 
            self.vtx_uv = vtx_uv #uv_coords
            self.uv_idx = faces.textures_idx.to(device).to(torch.int32).contiguous() #uv index

        else:
            print("generate tex coords")
            self.generate_uvs()
    
    def generate_uvs(self):
        v_np = self.vtx_pos.cpu().numpy()
        f_np = self.pos_idx.int().cpu().numpy()
        atlas = xatlas.Atlas()
        atlas.add_mesh(v_np, f_np)
        chart_options = xatlas.ChartOptions()
        chart_options.max_iterations = 4
        atlas.generate(chart_options=chart_options)
        vmapping, ft_np, vt_np = atlas[0]  # [N], [M, 3], [N, 2]

        vt = torch.from_numpy(vt_np.astype(np.float32)).float().cuda()
        ft = torch.from_numpy(ft_np.astype(np.int64)).int().cuda()
        
        self.vtx_uv = vt
        self.vtx_uv[:,1] = 1 - self.vtx_uv[:,1] 
        self.uv_idx = ft
    
    def save_obj(self, path,tex = None):
        vtx_uv = self.vtx_uv.clone() #uv_coords
        vtx_uv[:,1] = 1 - vtx_uv[:,1] 
        if tex is None:
            tex = self.tex
        from pytorch3d.io import save_obj
        save_obj(path, self.vtx_pos, self.pos_idx, verts_uvs = vtx_uv, faces_uvs = self.uv_idx, texture_map  = tex)
    
    def get_seg_faces(self, bbox_min, bbox_max):
        def points_in_bbox(points, bbox_min, bbox_max):
            # Check if points are greater than or equal to the minimum corner and
            # less than or equal to the maximum corner.
            in_bbox = (points >= bbox_min) & (points <= bbox_max)
            
            # All three coordinates should be inside the bounding box
            return in_bbox.all(dim=-1)

        points = self.vtx_pos #[V, 3]
        inside = points_in_bbox(points, bbox_min, bbox_max) #[V]
        faces = self.pos_idx.long()
        inside_faces = inside[faces]#[F, 3]
        inside_faces = inside_faces.any(dim = -1)[:, None].expand(-1, 3)

        inside_faces = inside_faces.to(torch.int32) 
        
        return inside_faces

    def rasterize(self, 
                glctx, 
                r_mvp_all, 
                rendersettings):

        pos_idx = self.pos_idx

        pos_clip = transform_pos_batched(r_mvp_all, self.vtx_pos[None,])
        rast_out, rast_out_db = dr.rasterize(glctx, pos_clip, pos_idx, resolution=[rendersettings.resolution, rendersettings.resolution])
        return rast_out, rast_out_db #[b, H, W, 4], [b, H, W, 4]

    def texture(self, 
                glctx,
                rast_out,
                rast_out_db,
                rendersettings,
                tex = None):
        
        uv = self.vtx_uv
        uv_idx = self.uv_idx
        if tex is None:
            tex = self.tex
        
        texc, texd = dr.interpolate(uv[None, ...], rast_out, uv_idx, rast_db=rast_out_db, diff_attrs='all')
        # print(rendersettings.interp_type)
        if rendersettings.enable_mip or rendersettings.interp_type == "mip":
            # print("mip map enabled")
            if rendersettings.absolute_bias_level and rendersettings.mip_level_bias is not None:
                color = dr.texture(tex[None, ...], texc, None, filter_mode='linear-mipmap-linear', max_mip_level=rendersettings.max_mip_level, mip_level_bias = rendersettings.mip_level_bias)
            else:
                color = dr.texture(tex[None, ...], texc, texd, filter_mode='linear-mipmap-linear', max_mip_level=rendersettings.max_mip_level, mip_level_bias = rendersettings.mip_level_bias)
        elif (rendersettings.interp_type == "linear") or (rendersettings.interp_type == "nearest") or (rendersettings.interp_type == "linear-gp"):
            color = dr.texture(tex[None, ...], texc, filter_mode=rendersettings.interp_type)
        else:
            raise Exception("unknown interpolation type (nearest, linear, linear-gp, or mip)")
    
        # color = dr.texture(tex[None, ...], texc, filter_mode="linear")
        if rendersettings.latent:
            color = color.permute(0, 3, 1, 2) #[b, C, H, W]      
        return color
    
    def render(self, 
                glctx, 
                r_mvp_all,
                rendersettings,
                tex = None):
        uv = self.vtx_uv
        uv_idx = self.uv_idx
        pos_idx = self.pos_idx
        if tex is None:
            tex = self.tex

        rast_out, rast_out_db = self.rasterize(glctx, r_mvp_all, rendersettings)
        
        color = self.texture(glctx, rast_out, rast_out_db, rendersettings, tex = tex)
        

        return color, rast_out #[b, H, W, C], [b, H, W, 4]
    
    def rasterize_all (self, glctx, view_dataset, rendersettings):
        r_mvp_all = view_dataset.r_mvp_all
        rast_out_all, rast_out_db_all = self.rasterize(glctx, r_mvp_all, rendersettings)   #[b, 64, 64, 4]         
        depth_map_all = rast_out_all[..., -2] #[b, 64, 64]
        
        object_mask_all = (depth_map_all  != 0)  #[b, 64, 64]
        if rendersettings.latent:
            rgb_mask_all = object_mask_all[:, None,:,:].expand(-1, rendersettings.channels, -1,-1) #[b, 4, H, W] 
        else:
            rgb_mask_all = object_mask_all[:,:,:, None].expand(-1, -1,-1,  rendersettings.channels) #[b, H, W, 3] 
        
        depth_mask_all = torch.zeros_like(depth_map_all[:,None])

        for j in range(view_dataset.len):
            depth_mask = normalize_depth(-depth_map_all[j])[None,]

            depth_mask = 2.0 * (depth_mask - depth_mask.min()) / (depth_mask.max() - depth_mask.min()) - 1.0
            depth_mask_all[j:j + 1] = depth_mask[None, ]
            # depth_mask_all[j:j + 1] = F.interpolate(depth_mask.unsqueeze(1), size=(64,64), mode='bicubic',
            #                         align_corners=False)

        return rast_out_all, object_mask_all, rgb_mask_all, depth_mask_all, rast_out_db_all
    
    
    def optimize_texture(self, 
                         glctx, 
                         img,
                         rendersettings,
                         r_mvp = None,
                         rgb_mask = None,
                         tex_opt = None,
                         tex_opt_shape= None, 
                         view_dataset = None,
                         keep_count = False,
                         clamp = False,
                         max_iter = 200,
                         lr_base = 1e-2):
        
        if (r_mvp is None) or (rgb_mask is None):
            if view_dataset is None:
                raise Exception("there should be a view input")
            r_mvp = view_dataset.r_mvp_all
            _, _, rgb_mask, _, _ = self.rasterize_all(glctx, view_dataset, rendersettings)
        
        if (tex_opt is None):
            if (tex_opt_shape is None):
                tex_opt = torch.full(self.tex.shape, 0.0, device=self.device, requires_grad=True)
            if tex_opt_shape is not None:
                tex_opt = torch.full(tex_opt_shape, 0.0, device=self.device, requires_grad=True)
        
        lr_ramp  = 0.1

        optimizer    = torch.optim.Adam([tex_opt], lr=lr_base)
        scheduler    = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: lr_ramp**(float(x)/float(max_iter)))
        
        rast_out, rast_out_db = self.rasterize(glctx, r_mvp, rendersettings)
        
        for it in range(max_iter + 1):
            optimizer.zero_grad()
            color_opt = self.texture(glctx, rast_out, rast_out_db, rendersettings, tex = tex_opt)
            # color_opt, rast_out_opt = self.render(glctx, r_mvp, rendersettings, tex = tex_opt)
            loss_rgb = ((color_opt - img) ** 2)[rgb_mask].mean()
            loss_rgb.backward()
            optimizer.step()
            if clamp:
                with torch.no_grad():    
                    tex_opt = tex_opt.clamp_(0, 1)

        del optimizer, color_opt, scheduler, loss_rgb
        torch.cuda.empty_cache()

        if keep_count:
            count = torch.any(tex_opt.detach() != 0, dim = 2).float()[:,:, None]

            return tex_opt, count
        
        return tex_opt
    
    def optimize_texture_aggregate(self, 
                     glctx, 
                     img_all,
                     rendersettings,
                     r_mvp_all = None,
                     rgb_mask_all = None,
                     view_dataset = None):
        tex_next = torch.zeros_like(self.tex)
        count_next = torch.zeros_like(self.tex[:,:, 0:1])
        
        if (r_mvp_all is None) or (rgb_mask_all is None):
            if view_dataset is None:
                raise Exception("there should be a view input")
            r_mvp_all = view_dataset.r_mvp_all
            _, _, rgb_mask_all, _, _ = self.rasterize_all(glctx, view_dataset, rendersettings)
        
        num_views = r_mvp_all.shape[0]
        counts = torch.zeros((r_mvp_all.shape[0], self.tex.shape[0], self.tex.shape[1], 1), dtype = torch.float32, device = "cuda")
        
        
        for j in range(num_views):
            
            tex_opt, count= self.optimize_texture(glctx,
                                            img_all[j: j + 1],
                                            rendersettings,
                                            r_mvp_all[j:j + 1],
                                            rgb_mask_all[j: j + 1],
                                            keep_count=True
                                           )
            
            tex_next = tex_next + tex_opt.detach()
            counts[j] = count
            count_next = count_next + counts[j] 
            del tex_opt
            torch.cuda.empty_cache()
        
        return tex_next, count_next, counts
    