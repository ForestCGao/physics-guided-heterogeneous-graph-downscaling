import numpy as np
import torch
from scipy.ndimage import label

class HeteroWeatherGraphBuilder:

    def __init__(self, grid_size=32, core_threshold=0.286, expand_threshold=0.143, min_area=2, min_core_area=1, k_min=4, k_max=12, bb_near_px=8, bb_wind_px=8, bb_wind_speed_thr=0.05, bb_cos_thr=0.5, bb_dbz_diff_near=0.2, bb_dbz_diff_wind=0.25):
        self.grid_size = grid_size
        self.core_threshold = core_threshold
        self.expand_threshold = expand_threshold
        self.min_area = min_area
        self.min_core_area = min_core_area
        self.k_min = k_min
        self.k_max = k_max
        self.num_A = grid_size * grid_size
        self.bb_near_px = bb_near_px
        self.bb_wind_px = bb_wind_px
        self.bb_wind_speed_thr = bb_wind_speed_thr
        self.bb_cos_thr = bb_cos_thr
        self.bb_dbz_diff_near = bb_dbz_diff_near
        self.bb_dbz_diff_wind = bb_dbz_diff_wind
        (y, x) = torch.meshgrid(torch.linspace(-1, 1, grid_size), torch.linspace(-1, 1, grid_size), indexing='ij')
        self.pos_A = torch.stack([x.flatten(), y.flatten()], dim=-1)
        self.spatial_mask_A = self._build_A_space_mask(grid_size)

    def _build_A_space_mask(self, size, radius=3, stride=4):
        mask = torch.zeros((self.num_A, self.num_A), dtype=torch.bool)
        for i in range(size):
            for j in range(size):
                u = i * size + j
                (rmin, rmax) = (max(0, i - radius), min(size, i + radius + 1))
                (cmin, cmax) = (max(0, j - radius), min(size, j + radius + 1))
                for x in range(rmin, rmax):
                    for y in range(cmin, cmax):
                        mask[u, x * size + y] = True
        for x in range(0, size, stride):
            for y in range(0, size, stride):
                mask[:, x * size + y] = True
        return mask

    def extract_storm_nodes(self, frame_data):
        """
        Extract B nodes using core and expansion thresholds.
        Each B node has five features:
        [area_ratio, core_area_ratio, max_dbz, mean_dbz, dbz_std]
        """
        dbz = frame_data[..., 0]
        u_wind = frame_data[..., 1]
        v_wind = frame_data[..., 2]
        core_map = (dbz > self.core_threshold).astype(np.int32)
        expand_map = (dbz > self.expand_threshold).astype(np.int32)
        (core_labels, num_core) = label(core_map, structure=np.ones((3, 3), dtype=np.int32))
        (expand_labels, num_expand) = label(expand_map, structure=np.ones((3, 3), dtype=np.int32))
        b_features = []
        b_pos = []
        pixel_to_storm = {}
        storm_details = []
        current_b_idx = 0
        used_expand_ids = set()
        for core_idx in range(1, num_core + 1):
            core_mask = core_labels == core_idx
            core_area = int(core_mask.sum())
            if core_area < self.min_core_area:
                continue
            overlapped_expand_ids = np.unique(expand_labels[core_mask])
            overlapped_expand_ids = overlapped_expand_ids[overlapped_expand_ids > 0]
            if len(overlapped_expand_ids) == 0:
                continue
            best_expand_id = None
            best_expand_area = -1
            for ex_id in overlapped_expand_ids:
                ex_mask = expand_labels == ex_id
                ex_area = int(ex_mask.sum())
                if ex_area > best_expand_area:
                    best_expand_area = ex_area
                    best_expand_id = ex_id
            if best_expand_id in used_expand_ids:
                continue
            used_expand_ids.add(best_expand_id)
            target_mask = expand_labels == best_expand_id
            area = int(target_mask.sum())
            if area < self.min_area:
                continue
            (y_idx, x_idx) = np.where(target_mask)
            flat_indices = y_idx * self.grid_size + x_idx
            for idx in flat_indices:
                pixel_to_storm[int(idx)] = current_b_idx
            storm_dbz = dbz[target_mask]
            area_ratio = area / self.num_A
            core_area_ratio = core_area / (area + 1e-06)
            max_dbz = float(storm_dbz.max())
            mean_dbz = float(storm_dbz.mean())
            dbz_std = float(storm_dbz.std())
            b_features.append([area_ratio, core_area_ratio, max_dbz, mean_dbz, dbz_std])
            pos_x = float(x_idx.mean() / (self.grid_size - 1) * 2 - 1)
            pos_y = float(y_idx.mean() / (self.grid_size - 1) * 2 - 1)
            b_pos.append([pos_x, pos_y])
            storm_u = float(u_wind[target_mask].mean())
            storm_v = float(v_wind[target_mask].mean())
            storm_details.append({'area_px': area, 'mean_u_norm': storm_u, 'mean_v_norm': storm_v})
            current_b_idx += 1
        if len(b_features) > 0:
            nf_B = torch.tensor(b_features, dtype=torch.float32)
            pos_B = torch.tensor(b_pos, dtype=torch.float32)
        else:
            nf_B = torch.empty((0, 5), dtype=torch.float32)
            pos_B = torch.empty((0, 2), dtype=torch.float32)
        frame_stats = {'storm_details': storm_details}
        return (nf_B, pos_B, pixel_to_storm, frame_stats)

    def build_frame_graph(self, frame_data, time_step, current_offset):
        nf_A = torch.tensor(frame_data.reshape(-1, 5), dtype=torch.float32)
        pos_A = self.pos_A
        type_A = torch.zeros(self.num_A, dtype=torch.long)
        (nf_B, pos_B, pixel_to_storm, frame_stats) = self.extract_storm_nodes(frame_data)
        num_B = nf_B.shape[0]
        type_B = torch.ones(num_B, dtype=torch.long)
        nf_all = torch.cat([nf_A, nf_B], dim=0)
        pos_all = torch.cat([pos_A, pos_B], dim=0)
        type_all = torch.cat([type_A, type_B], dim=0)
        dist_A = torch.cdist(nf_A, nf_A, p=2)
        dist_A.masked_fill_(~self.spatial_mask_A, float('inf'))
        dist_A.fill_diagonal_(float('inf'))
        grid_features = nf_A.view(self.grid_size, self.grid_size, 5).permute(2, 0, 1)
        (grad_y, grad_x) = torch.gradient(grid_features, dim=(1, 2))
        grad_mag_all = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-08)
        grad_mag_weighted = 1.0 * grad_mag_all[0] + 0.5 * grad_mag_all[1] + 0.5 * grad_mag_all[2]
        grad_mag_flat = grad_mag_weighted.flatten()
        grad_norm = (grad_mag_flat - grad_mag_flat.min()) / (grad_mag_flat.max() - grad_mag_flat.min() + 1e-06)
        degrees = (self.k_min + (self.k_max - self.k_min) * grad_norm).long()
        (_, topk_indices) = torch.topk(dist_A, self.k_max, dim=1, largest=False)
        col_indices = torch.arange(self.k_max).expand(self.num_A, self.k_max)
        valid_mask = col_indices < degrees.unsqueeze(1)
        src_nodes = torch.arange(self.num_A).unsqueeze(1).expand(self.num_A, self.k_max)
        edge_src_0 = src_nodes[valid_mask]
        edge_dst_0 = topk_indices[valid_mask]
        edge_type_0 = torch.zeros_like(edge_src_0)
        if len(pixel_to_storm) > 0:
            p_indices = torch.tensor(list(pixel_to_storm.keys()), dtype=torch.long)
            b_indices = torch.tensor(list(pixel_to_storm.values()), dtype=torch.long) + self.num_A
            (edge_src_1, edge_dst_1) = (p_indices, b_indices)
            (edge_src_2, edge_dst_2) = (b_indices, p_indices)
            edge_type_1 = torch.ones_like(edge_src_1)
            edge_type_2 = torch.full_like(edge_src_2, 2)
        else:
            edge_src_1 = edge_dst_1 = edge_type_1 = torch.empty(0, dtype=torch.long)
            edge_src_2 = edge_dst_2 = edge_type_2 = torch.empty(0, dtype=torch.long)
        (edge_src_3_list, edge_dst_3_list, edge_type_3_list) = ([], [], [])
        pixel_scale = 2.0 / (self.grid_size - 1)
        if num_B > 1:
            for i in range(num_B):
                for j in range(num_B):
                    if i == j:
                        continue
                    b_i_global = self.num_A + i
                    b_j_global = self.num_A + j
                    vec_pos = pos_B[j] - pos_B[i]
                    dist_pos = torch.norm(vec_pos).item()
                    storm_i = frame_stats['storm_details'][i]
                    storm_j = frame_stats['storm_details'][j]
                    vec_wind = torch.tensor([storm_i['mean_u_norm'], storm_i['mean_v_norm']], dtype=torch.float32)
                    wind_speed = torch.norm(vec_wind).item()
                    area_i_px = storm_i['area_px']
                    area_j_px = storm_j['area_px']
                    dbz_i = nf_B[i, 3].item()
                    dbz_j = nf_B[j, 3].item()
                    dbz_diff = abs(dbz_i - dbz_j)
                    near_px_i = self.bb_near_px + 0.5 * np.sqrt(area_i_px)
                    near_px_j = self.bb_near_px + 0.5 * np.sqrt(area_j_px)
                    near_px_ij = max(near_px_i, near_px_j)
                    wind_px_i = self.bb_wind_px + 0.8 * np.sqrt(area_i_px)
                    wind_px_j = self.bb_wind_px + 0.8 * np.sqrt(area_j_px)
                    wind_px_ij = max(wind_px_i, wind_px_j)
                    near_dist = near_px_ij * pixel_scale
                    wind_dist = wind_px_ij * pixel_scale
                    connect = False
                    vec_wind_j = torch.tensor([storm_j['mean_u_norm'], storm_j['mean_v_norm']], dtype=torch.float32)
                    wind_speed_j = torch.norm(vec_wind_j).item()
                    i_to_j = False
                    if wind_speed > self.bb_wind_speed_thr:
                        cos_ij = torch.dot(vec_pos, vec_wind).item() / (dist_pos * wind_speed + 1e-06)
                        if cos_ij > self.bb_cos_thr:
                            i_to_j = True
                    j_to_i = False
                    if wind_speed_j > self.bb_wind_speed_thr:
                        cos_ji = torch.dot(-vec_pos, vec_wind_j).item() / (dist_pos * wind_speed_j + 1e-06)
                        if cos_ji > self.bb_cos_thr:
                            j_to_i = True
                    if dist_pos < near_dist and dbz_diff < self.bb_dbz_diff_near:
                        if i_to_j and (not j_to_i):
                            connect = True
                        elif not i_to_j and j_to_i:
                            connect = False
                        else:
                            connect = True
                    elif wind_speed > self.bb_wind_speed_thr and 1e-05 < dist_pos < wind_dist:
                        cos_theta = torch.dot(vec_pos, vec_wind).item() / (dist_pos * wind_speed + 1e-06)
                        if cos_theta > self.bb_cos_thr and dbz_diff < self.bb_dbz_diff_wind:
                            connect = True
                    if connect:
                        edge_src_3_list.append(b_i_global)
                        edge_dst_3_list.append(b_j_global)
                        edge_type_3_list.append(3)
        edge_src_3 = torch.tensor(edge_src_3_list, dtype=torch.long)
        edge_dst_3 = torch.tensor(edge_dst_3_list, dtype=torch.long)
        edge_type_3 = torch.tensor(edge_type_3_list, dtype=torch.long)
        edge_src = torch.cat([edge_src_0, edge_src_1, edge_src_2, edge_src_3]) + current_offset
        edge_dst = torch.cat([edge_dst_0, edge_dst_1, edge_dst_2, edge_dst_3]) + current_offset
        edge_type = torch.cat([edge_type_0, edge_type_1, edge_type_2, edge_type_3])
        edge_time = torch.full_like(edge_type, time_step)
        return (nf_all, type_all, edge_src, edge_dst, edge_type, edge_time, pos_all, len(nf_all), frame_stats)

    def process_sequence(self, sequence_data):
        frames = sequence_data.shape[0]
        (node_features, node_types, all_pos_list) = ([], [], [])
        (e_src, e_dst, e_type, e_time) = ([], [], [], [])
        node_offsets = []
        current_offset = 0
        for t in range(frames):
            node_offsets.append(current_offset)
            frame_data = sequence_data[t]
            (nf, nt, src, dst, etype, etime, pos, num_nodes, frame_stats) = self.build_frame_graph(frame_data, time_step=t, current_offset=current_offset)
            node_features.append(nf)
            node_types.append(nt)
            all_pos_list.append(pos)
            e_src.append(src)
            e_dst.append(dst)
            e_type.append(etype)
            e_time.append(etime)
            current_offset += num_nodes
        node_offsets.append(current_offset)
        nf_all = torch.cat(node_features, dim=0)
        nt_all = torch.cat(node_types, dim=0)
        edge_index_all = torch.stack([torch.cat(e_src), torch.cat(e_dst)], dim=0)
        etype_all = torch.cat(e_type)
        etime_all = torch.cat(e_time)
        pos_all = torch.cat(all_pos_list)
        return (nf_all, nt_all, edge_index_all, etype_all, etime_all, node_offsets, pos_all)


def make_graph(sequence_data, **graph_settings):
    """Build a graph from normalized [T, S, S, 5] NumPy inputs.

    Node type 0 is G (called A in the source); type 1 is C (called B).
    Relations 0..3 are G->G, G->C, C->G, C->C.
    """
    sequence_data = np.asarray(sequence_data, dtype=np.float32)
    builder = HeteroWeatherGraphBuilder(**graph_settings)
    expected = (builder.grid_size, builder.grid_size, 5)
    if sequence_data.ndim != 4 or sequence_data.shape[1:] != expected:
        raise ValueError(f'Expected [T,{expected[0]},{expected[1]},5], got {sequence_data.shape}')
    if sequence_data.shape[0] < 1 or not np.isfinite(sequence_data).all():
        raise ValueError('Graph inputs must contain finite values and at least one frame.')
    nf, nt, ei, ety, et, offsets, pos = builder.process_sequence(sequence_data)
    return dict(nf=nf, nt=nt, ei=ei, ety=ety, et=et, pos=pos,
                times=torch.arange(len(sequence_data), dtype=torch.float32),
                offsets=torch.tensor(offsets, dtype=torch.long))
