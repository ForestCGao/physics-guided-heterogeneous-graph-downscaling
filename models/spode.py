import torch
from torch import nn
from torch.nn import functional as F
from torchdiffeq import odeint
from einops import rearrange

def _offset_to_int(x):
    if torch.is_tensor(x):
        return int(x.detach().item())
    return int(x)

def _time_to_int(x):
    if torch.is_tensor(x):
        return int(x.detach().item())
    return int(x)

def _time_to_float(x):
    if torch.is_tensor(x):
        return float(x.detach().item())
    return float(x)

class PhysicsODEFunc(nn.Module):

    def __init__(self, dim, num_A_nodes):
        super().__init__()
        self.advection_coupling_AA = nn.Parameter(torch.Tensor(1, dim))
        self.advection_coupling_AB = nn.Parameter(torch.Tensor(1, dim))
        self.advection_coupling_BA = nn.Parameter(torch.Tensor(1, dim))
        self.advection_coupling_BB = nn.Parameter(torch.Tensor(1, dim))
        nn.init.uniform_(self.advection_coupling_AA, -0.1, 0.1)
        nn.init.uniform_(self.advection_coupling_AB, -0.1, 0.1)
        nn.init.uniform_(self.advection_coupling_BA, -0.1, 0.1)
        nn.init.uniform_(self.advection_coupling_BB, -0.1, 0.1)
        self.time_dim = 8
        self.time_net = nn.Sequential(nn.Linear(1, self.time_dim), nn.Tanh())
        self.pos_dim = 8
        self.pos_emb_A = nn.Embedding(num_A_nodes, self.pos_dim)
        self.pos_map_B = nn.Sequential(nn.Linear(2, self.pos_dim), nn.GELU())
        input_size = dim + self.time_dim + self.pos_dim
        self.source_nets = nn.ModuleDict({'A': nn.Sequential(nn.Linear(input_size, dim * 2), nn.LayerNorm(dim * 2), nn.GELU(), nn.Linear(dim * 2, dim)), 'B': nn.Sequential(nn.Linear(input_size, dim * 2), nn.LayerNorm(dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))})
        self.edge_index = None
        self.attention_weights = None
        self.pos = None
        self.wind_uv = None
        self.node_type = None
        self.cached_pos_vec = None

    def set_env(self, edge_index, attention_weights, pos, wind_uv, node_type):
        self.edge_index = edge_index
        self.pos = pos
        self.wind_uv = wind_uv
        if node_type.dim() > 1:
            node_type = node_type.squeeze(-1)
        self.node_type = node_type.long()
        if attention_weights.dim() > 1:
            attention_weights = attention_weights.mean(dim=1)
        self.attention_weights = attention_weights
        num_nodes = pos.shape[0]
        device = pos.device
        self.cached_pos_vec = torch.zeros(num_nodes, self.pos_dim, dtype=pos.dtype, device=device)
        mask_A = self.node_type == 0
        mask_B = self.node_type == 1
        if mask_A.any():
            num_A_curr = int(mask_A.sum().item())
            a_ids = torch.arange(num_A_curr, device=device)
            self.cached_pos_vec[mask_A] = self.pos_emb_A(a_ids)
        if mask_B.any():
            b_coords = pos[mask_B]
            self.cached_pos_vec[mask_B] = self.pos_map_B(b_coords)

    def forward(self, t, h):
        (src, dst) = self.edge_index
        num_nodes = h.shape[0]
        assert src.min() >= 0
        assert dst.min() >= 0
        assert src.max() < num_nodes, f'src.max={src.max().item()}, num_nodes={num_nodes}'
        assert dst.max() < num_nodes, f'dst.max={dst.max().item()}, num_nodes={num_nodes}'
        src_type = self.node_type[src]
        dst_type = self.node_type[dst]
        diffusion_term = torch.zeros_like(h)
        alpha = self.attention_weights.view(-1, 1)
        diff_msg = alpha * (h[src] - h[dst])
        diffusion_term.index_add_(0, dst, diff_msg)
        advection_term = torch.zeros_like(h)
        rel_pos = self.pos[dst] - self.pos[src]
        dist = torch.norm(rel_pos, dim=1, keepdim=True).clamp(min=1e-06)
        dir_vec = rel_pos / dist
        wind_src = self.wind_uv[src]
        projection = (wind_src * dir_vec).sum(dim=1, keepdim=True)
        flow_strength = F.relu(projection)
        delta_h = h[src] - h[dst]
        mask_AA = (src_type == 0) & (dst_type == 0)
        mask_AB = (src_type == 0) & (dst_type == 1)
        mask_BA = (src_type == 1) & (dst_type == 0)
        mask_BB = (src_type == 1) & (dst_type == 1)
        if mask_AA.any():
            flux_AA = flow_strength[mask_AA] * delta_h[mask_AA] * self.advection_coupling_AA
            advection_term.index_add_(0, dst[mask_AA], flux_AA)
        if mask_AB.any():
            flux_AB = flow_strength[mask_AB] * delta_h[mask_AB] * self.advection_coupling_AB
            advection_term.index_add_(0, dst[mask_AB], flux_AB)
        if mask_BA.any():
            flux_BA = flow_strength[mask_BA] * delta_h[mask_BA] * self.advection_coupling_BA
            advection_term.index_add_(0, dst[mask_BA], flux_BA)
        if mask_BB.any():
            flux_BB = flow_strength[mask_BB] * delta_h[mask_BB] * self.advection_coupling_BB
            advection_term.index_add_(0, dst[mask_BB], flux_BB)
        t_vec = self.time_net(t.view(1, 1)).expand(h.shape[0], -1)
        g_input = torch.cat([h, t_vec, self.cached_pos_vec], dim=-1)
        source_term = torch.zeros_like(h)
        mask_A = self.node_type == 0
        mask_B = self.node_type == 1
        if mask_A.any():
            source_term[mask_A] = self.source_nets['A'](g_input[mask_A])
        if mask_B.any():
            source_term[mask_B] = self.source_nets['B'](g_input[mask_B])
        return diffusion_term + advection_term + source_term

class HGAD_ODE_Model(nn.Module):

    def __init__(self, gnn, hidden_dim, num_A, solver='euler'):
        super().__init__()
        self.gnn = gnn
        self.num_A = num_A
        self.solver = solver
        self.ode_func = PhysicsODEFunc(hidden_dim, num_A_nodes=num_A)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.norm_obs = nn.LayerNorm(hidden_dim)
        self.norm_pred = nn.LayerNorm(hidden_dim)

    def forward(self, nf, nt, et, ei, ety, node_pos, times, node_offsets):
        device = nf.device
        (z, all_states, all_attentions) = self.gnn(nf, nt, et, ei, ety, return_all_states=True)
        full_att = None
        if hasattr(self.gnn, 'gcs'):
            for i in reversed(range(len(self.gnn.gcs))):
                key = f'layer_{i}'
                if key in all_attentions and all_attentions[key] is not None:
                    full_att = all_attentions[key]
                    break
        if full_att is None:
            full_att = torch.ones(ei.shape[1], 1, device=device)
        if full_att.dim() > 1:
            full_att = full_att.mean(dim=1)
        wind_field = nf[:, 1:3]
        start0 = _offset_to_int(node_offsets[0])
        end0 = _offset_to_int(node_offsets[1])
        h_current_all = z[start0:end0]
        latent_history_A = [h_current_all[:self.num_A]]
        hgt_history_A = [h_current_all[:self.num_A]]
        for i in range(len(times) - 1):
            t_now = times[i]
            t_next = times[i + 1]
            t_now_int = _time_to_int(t_now)
            t_now_float = _time_to_float(t_now)
            t_next_float = _time_to_float(t_next)
            curr_start = _offset_to_int(node_offsets[i])
            curr_end = _offset_to_int(node_offsets[i + 1])
            next_start = _offset_to_int(node_offsets[i + 1])
            next_end = _offset_to_int(node_offsets[i + 2])
            edge_mask = et == t_now_int
            sub_edges = ei[:, edge_mask] - curr_start
            sub_att = full_att[edge_mask]
            sub_pos = node_pos[curr_start:curr_end]
            sub_wind = wind_field[curr_start:curr_end]
            sub_nt = nt[curr_start:curr_end]
            self.ode_func.set_env(edge_index=sub_edges, attention_weights=sub_att, pos=sub_pos, wind_uv=sub_wind, node_type=sub_nt)
            t_span = torch.tensor([t_now_float, t_next_float], dtype=torch.float32, device=device)
            original_dtype = z.dtype
            with torch.cuda.amp.autocast(enabled=False):
                h_current_f32 = h_current_all.to(torch.float32)
                h_pred_all_f32 = odeint(self.ode_func, h_current_f32, t_span, method=self.solver)[1]
            h_pred_all = h_pred_all_f32.to(original_dtype)
            h_obs_all = z[next_start:next_end]
            h_pred_A = self.norm_pred(h_pred_all[:self.num_A])
            h_obs_A = self.norm_obs(h_obs_all[:self.num_A])
            h_updated_A = self.gru(h_obs_A, h_pred_A)
            latent_history_A.append(h_updated_A)
            hgt_history_A.append(h_obs_all[:self.num_A])
            h_next_B = h_obs_all[self.num_A:]
            h_current_all = torch.cat([h_updated_A, h_next_B], dim=0)
        latent_history_A = torch.stack(latent_history_A, dim=0)
        hgt_history_A = torch.stack(hgt_history_A, dim=0)
        return (latent_history_A, hgt_history_A)

class PixelDecoderHeadNoBatch02(nn.Module):

    def __init__(self, in_dim):
        super().__init__()
        mid_dim = max(in_dim // 2, 1)
        low_dim = max(in_dim // 4, 1)
        self.block1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(in_dim, mid_dim, kernel_size=3, padding=1), nn.GELU(), nn.Conv2d(mid_dim, mid_dim, kernel_size=3, padding=1), nn.GELU())
        self.block2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(mid_dim, low_dim, kernel_size=3, padding=1), nn.GELU(), nn.Conv2d(low_dim, low_dim, kernel_size=3, padding=1), nn.GELU())
        self.out = nn.Conv2d(low_dim, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.out(x)
        x = torch.tanh(x)
        return x.squeeze(1)


class STRepresentationNetHRNoBatch(nn.Module):
    """Final weighted HGT + ODE/GRU model with the paper CNN decoder."""

    def __init__(self, gnn_model, hidden_dim, num_A_side, solver='euler',
                 residual_alpha_init=0.1):
        super().__init__()
        self.num_A_side = num_A_side
        self.num_A = num_A_side * num_A_side
        self.hidden_dim = hidden_dim
        self.hgad_model = HGAD_ODE_Model(
            gnn=gnn_model, hidden_dim=hidden_dim, num_A=self.num_A, solver=solver,
        )
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.pixel_decoder = PixelDecoderHeadNoBatch02(hidden_dim)
        self.residual_alpha = nn.Parameter(
            torch.tensor(float(residual_alpha_init), dtype=torch.float32)
        )

    def forward(self, nf, nt, et, ei, ety, node_pos, times, node_offsets):
        """Return [T,4*S,4*S], or [8,128,128] with the paper defaults."""
        latents_ode_gru, latents_hgt = self.hgad_model(
            nf=nf, nt=nt, et=et, ei=ei, ety=ety, node_pos=node_pos,
            times=times, node_offsets=node_offsets,
        )
        alpha = self.residual_alpha.to(
            device=latents_ode_gru.device, dtype=latents_ode_gru.dtype,
        )
        latents = latents_hgt + alpha * latents_ode_gru
        latents = self.out_proj(latents)
        grid_latents = rearrange(
            latents, 't (h w) c -> t c h w',
            h=self.num_A_side, w=self.num_A_side,
        )
        return self.pixel_decoder(grid_latents)
