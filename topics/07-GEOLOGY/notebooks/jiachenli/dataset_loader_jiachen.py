# Dataset loader
import os
import gc
import rasterio
import numpy as np

# --------------------- Reading and procssing functions ---------------------
def _window_mean2d(x, win_h, win_w, step_h, step_w):
    H, W = x.shape

    row_starts = np.arange(0, H - win_h + 1, step_h)
    col_starts = np.arange(0, W - win_w + 1, step_w)

    out = np.empty((len(row_starts), len(col_starts)), dtype=x.dtype)

    for i, r0 in enumerate(row_starts):
        r1 = r0 + win_h
        for j, c0 in enumerate(col_starts):
            c1 = c0 + win_w
            out[i, j] = x[r0:r1, c0:c1].mean()

    return out

"""
Multi-looking and calculating covariance matrix in PolSARpro style
    Input: S ([T, C, H, W]) - original scattering matrix, can be multiple views
           win ([La, Lr]) - window length in size of [azimuth, range]
    Output: C ([T, H, W, 3, 3]) - covariance matrix
"""
def multilook_C_T(S_data, win_size, stride_size=None):
    S_data = np.asarray(S_data)

    C_list = []
    for t in range(S_data.shape[0]):
        C_t = _multilook_single_scene(S_data[t], win_size, stride_size)
        C_list.append(C_t)
    return np.stack(C_list, axis=0)

def _multilook_single_scene(S, win_size, stride_size=None):
    La, Lr = win_size

    if stride_size is None:
        Sa, Sr = La, Lr
    else:
        Sa, Sr = stride_size

    C_in, H, W = S.shape

    if C_in == 4:
        Shh, Shv, Svh, Svv = S
        Shv = 0.5 * (Shv + Svh)
    elif C_in == 3:
        Shh, Shv, Svv = S
    else:
        raise ValueError("Input S must have 3 or 4 channels")

    # Lexicographic basis
    kl1 = Shh
    kl2 = np.sqrt(2) * Shv
    kl3 = Svv
    C11 = _window_mean2d(kl1 * np.conj(kl1), La, Lr, Sa, Sr)
    C12 = _window_mean2d(kl1 * np.conj(kl2), La, Lr, Sa, Sr)
    C13 = _window_mean2d(kl1 * np.conj(kl3), La, Lr, Sa, Sr)
    C22 = _window_mean2d(kl2 * np.conj(kl2), La, Lr, Sa, Sr)
    C23 = _window_mean2d(kl2 * np.conj(kl3), La, Lr, Sa, Sr)
    C33 = _window_mean2d(kl3 * np.conj(kl3), La, Lr, Sa, Sr)

    H_out, W_out = C11.shape
    C = np.zeros((H_out, W_out, 3, 3), dtype=np.complex64)

    C[..., 0, 0] = C11
    C[..., 0, 1] = C12
    C[..., 0, 2] = C13
    C[..., 1, 1] = C22
    C[..., 1, 2] = C23
    C[..., 2, 2] = C33
    C[..., 1, 0] = np.conj(C[..., 0, 1])
    C[..., 2, 0] = np.conj(C[..., 0, 2])
    C[..., 2, 1] = np.conj(C[..., 1, 2])

    return C.astype(np.complex64)

"""
Read complex SLC (float32 real/imag interleaved)
"""
def read_slc(filepath, H, W, dtype=np.float32):
    data = np.fromfile(filepath, dtype=dtype)
    data = data.reshape(H, W, 2)

    real = data[..., 0].astype(np.float32)
    imag = data[..., 1].astype(np.float32)

    complex_data = real + 1j * imag

    return complex_data.astype(np.complex64)

def apply_spatial_crop(S, crop_cfg):
    if not crop_cfg or not crop_cfg.get("enabled", False):
        return S

    r0 = crop_cfg.get("row_start", None)
    r1 = crop_cfg.get("row_end", None)
    c0 = crop_cfg.get("col_start", None)
    c1 = crop_cfg.get("col_end", None)

    S = S[:, r0:r1, c0:c1]

    return np.ascontiguousarray(S)

# ------------------- Reading and loading data -------------------
# Unified interface base
class SARdataLoader:
    def __init__(self, cfg_dataset: dict):
        self.cfg_dataset = cfg_dataset
        self.config = cfg_dataset["cfg_read"]
        self.win_size = cfg_dataset["win_size"]
        self.stride_size = cfg_dataset.get("stride_size", self.win_size)

    def load_one_scene(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement load_one_scene()")

    def load(self):
        raise NotImplementedError("Subclasses must implement load()")
    
    def pre_process(self, S):
        if self.config.get("flip_ud", False):
            S = np.flip(S, axis=1)

        if self.config.get("flip_lr", False):
            S = np.flip(S, axis=2)

        S = apply_spatial_crop(S, self.config.get("crop", None))
        S = np.ascontiguousarray(S).astype(np.complex64)

        C_t = multilook_C_T(np.expand_dims(S, axis=0), self.win_size, self.stride_size)[0]
        return C_t.astype(np.complex64)
    
    def pack_C_QP(self, C_list, scene_ids):
        C_QP = np.stack(C_list, axis=0).astype(np.complex64)
        return {
            "C_QP": C_QP,
            "scene_ids": scene_ids}

# Biomass (single view or multiple view) loader
class BiomassLoader(SARdataLoader):
    def _scene_id_from_name(self, scene_name):
        for part in scene_name.split("_"):
            if len(part) == 15 and "T" in part:
                return part
        return scene_name
    
    def discover_scene_dirs(self, root_dir):
        scene_dirs = []

        for name in os.listdir(root_dir):
            scene_path = os.path.join(root_dir, name)
            measurement_path = os.path.join(scene_path, "measurement")

            if os.path.isdir(measurement_path):
                scene_id = self._scene_id_from_name(name)
                scene_dirs.append((scene_id, measurement_path))

        if not scene_dirs:
            raise FileNotFoundError(f"No BIOMASS scene measurement folders found in {root_dir}")

        scene_dirs.sort(key=lambda x: x[0])
        return scene_dirs

    def discover_tiff_pair(self, path_input):
        abs_files = [f for f in os.listdir(path_input) if f.lower().endswith("_i_abs.tiff")]
        phase_files = [f for f in os.listdir(path_input) if f.lower().endswith("_i_phase.tiff")]

        if len(abs_files) != 1 or len(phase_files) != 1:
            raise FileNotFoundError(
                f"Expected one *_i_abs.tiff and one *_i_phase.tiff in {path_input},"
                f"got abs={abs_files}, phase={phase_files}")

        return abs_files[0], phase_files[0]

    def load_one_scene(self, path_input=None, tiff_abs=None, tiff_phase=None):
        if path_input is None:
            path_input = self.config["path_input"]
        if tiff_abs is None or tiff_phase is None:
            tiff_abs, tiff_phase = self.discover_tiff_pair(path_input)

        # Read raw data 
        with rasterio.open(os.path.join(path_input, tiff_abs)) as ds_abs:
            data_abs = ds_abs.read()
        with rasterio.open(os.path.join(path_input, tiff_phase)) as ds_phase:
            data_phase = ds_phase.read()

        if data_abs.shape[0] != 4:
            print(f"[Warning] Data has {data_abs.shape[0]} channels, not full-pol.")

        # Extract channels
        A_hh, A_hv, A_vh, A_vv = data_abs
        phi_hh, phi_hv, phi_vh, phi_vv = data_phase

        # Construct complex SLC
        S_hh = A_hh * np.exp(1j * phi_hh)
        S_hv = A_hv * np.exp(1j * phi_hv)
        S_vh = A_vh * np.exp(1j * phi_vh)
        S_vv = A_vv * np.exp(1j * phi_vv)

        return np.stack([S_hh, S_hv, S_vh, S_vv], axis=0).astype(np.complex64)

    def load(self):
        C_list = []
        scene_ids = []

        if "path_input" in self.config:
            scene_id = self.config.get("scene_id", "biomass")
            scenes = [(scene_id, self.config["path_input"])]
        elif "path_root" in self.config:
            scenes = self.discover_scene_dirs(self.config["path_root"])
        else:
            raise ValueError("BiomassLoader requires either 'path_input' or 'path_root' in config.")

        for scene_id, path_input in scenes:
            S_t = self.load_one_scene(path_input)
            C_t = self.pre_process(S_t)

            C_list.append(C_t)
            scene_ids.append(scene_id)

            del S_t
            gc.collect()

        print(f"[BIOMASS] Loaded C_QP with {len(scene_ids)} scene")
        print("Suggestted multi-look window size [12, 2]")

        return self.pack_C_QP(C_list, scene_ids)

# RadarSat-2 (single view or multiple view) loader
class RadarSat2Loader(SARdataLoader):
    def load_one_scene(self, scene_path, rows, cols):
        hh_file = None
        for f in os.listdir(scene_path):
            if f.endswith("_slc_HH.dat"):
                hh_file = f
                break

        if hh_file is None:
            raise FileNotFoundError(f"No _slc_HH.dat file found in {scene_path}")

        prefix = hh_file.replace("_slc_HH.dat", "")

        hv_file = prefix + "_slc_HV.dat"
        vh_file = prefix + "_slc_VH.dat"
        vv_file = prefix + "_slc_VV.dat"

        hh_path = os.path.join(scene_path, hh_file)
        hv_path = os.path.join(scene_path, hv_file)
        vh_path = os.path.join(scene_path, vh_file)
        vv_path = os.path.join(scene_path, vv_file)

        S_hh = read_slc(hh_path, rows, cols)
        S_hv = read_slc(hv_path, rows, cols)
        S_vh = read_slc(vh_path, rows, cols)
        S_vv = read_slc(vv_path, rows, cols)

        S = np.stack([S_hh, S_hv, S_vh, S_vv], axis=0).astype(np.complex64)

        return S

    # Output a list of [folder_name, path_input]
    def discover_scene_dirs(self, root_dir, folder_suffix):
        scene_dirs = []
        for name in os.listdir(root_dir):
            full_path = os.path.join(root_dir, name)
            if os.path.isdir(full_path) and name.endswith(folder_suffix):
                scene_dirs.append((name, full_path))
        if not scene_dirs:
            raise FileNotFoundError(f"No scene folders ending with {folder_suffix} found in {root_dir}")
        
        scene_dirs.sort(key=lambda x: x[0])
        return scene_dirs

    def load(self):
        rows = self.config["rows"]
        cols = self.config["cols"]

        C_list = []
        scene_ids = []

        if "path_input" in self.config:
            scene_path = self.config["path_input"]
            scene_name = os.path.basename(os.path.normpath(scene_path))
            scenes = [(scene_name.replace(".rst2.qp", ""), scene_path)]
        elif "path_root" in self.config:
            root_dir = self.config["path_root"]
            folder_suffix = self.config.get("folder_suffix", ".rst2.qp")
            scenes = [(scene_name.replace(".rst2.qp", ""), scene_path) 
                      for scene_name, scene_path in self.discover_scene_dirs(root_dir, folder_suffix)]
        else:
            raise ValueError(
                "RadarSat2Loader requires either 'path_input' or 'path_root' in config.")

        for scene_id, scene_path in scenes:
            S_t = self.load_one_scene(scene_path, rows, cols)
            C_t = self.pre_process(S_t)

            C_list.append(C_t)
            scene_ids.append(scene_id)

            del S_t
            gc.collect()

        print(f"[RadarSat-2] Loaded C_QP with {len(scene_ids)} scenes")
        print("Suggestted multi-look window size [4, 2]")

        return self.pack_C_QP(C_list, scene_ids)

class SAOCOMLoader(SARdataLoader):
    POL_ORDER = ("HH", "HV", "VH", "VV")

    def load_one_scene(self, root_dir, scene_id, rows, cols):
        hh_file = f"{scene_id}_HH.SLC"
        hv_file = f"{scene_id}_HV.SLC"
        vh_file = f"{scene_id}_VH.SLC"
        vv_file = f"{scene_id}_VV.SLC"

        hh_path = os.path.join(root_dir, hh_file)
        hv_path = os.path.join(root_dir, hv_file)
        vh_path = os.path.join(root_dir, vh_file)
        vv_path = os.path.join(root_dir, vv_file)

        for path in [hh_path, hv_path, vh_path, vv_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing SAOCOM SLC file: {path}")

        S_hh = read_slc(hh_path, rows, cols, dtype=np.int16)
        S_hv = read_slc(hv_path, rows, cols, dtype=np.int16)
        S_vh = read_slc(vh_path, rows, cols, dtype=np.int16)
        S_vv = read_slc(vv_path, rows, cols, dtype=np.int16)

        S = np.stack([S_hh, S_hv, S_vh, S_vv], axis=0).astype(np.complex64)

        return S

    def discover_scene_ids(self, root_dir):
        scene_map = {}

        for name in os.listdir(root_dir):
            if not name.endswith(".SLC"):
                continue

            stem = name.replace(".SLC", "")
            parts = stem.split("_")
            sat, date, pol = parts
            scene_id = f"{sat}_{date}"

            if scene_id not in scene_map:
                scene_map[scene_id] = set()

            scene_map[scene_id].add(pol)

        scene_ids = []
        for scene_id, pols in scene_map.items():
            if all(pol in pols for pol in self.POL_ORDER):
                scene_ids.append(scene_id)

        if not scene_ids:
            raise FileNotFoundError(f"No complete SAOCOM full-pol scenes found in {root_dir}")

        scene_ids.sort()

        return scene_ids

    def load(self):
        root_dir = self.config["path_root"]
        rows = self.config["rows"]
        cols = self.config["cols"]

        C_list = []
        scene_ids = []

        if "scene_id" in self.config:
            scenes = [self.config["scene_id"]]
        elif "scene_ids" in self.config:
            scenes = self.config["scene_ids"]
        else:
            scenes = self.discover_scene_ids(root_dir)

        for scene_id in scenes:
            S_t = self.load_one_scene(root_dir, scene_id, rows, cols)
            C_t = self.pre_process(S_t)

            C_list.append(C_t)
            scene_ids.append(scene_id)

            del S_t
            gc.collect()

        print(f"[SAOCOM] Loaded C_QP with {len(scene_ids)} scenes")
        print("Suggested multi-look window size [12, 4]")

        return self.pack_C_QP(C_list, scene_ids)

def get_loader(cfg_dataset: dict) -> SARdataLoader:
    dataset_type = cfg_dataset["dataset_type"].lower()

    if dataset_type in ("biomass", "biomass_timeseries"):
        return BiomassLoader(cfg_dataset)
    elif dataset_type in ("radarsat2", "radarsat2_timeseries", "radarsat2_timeseries_clean"):
        return RadarSat2Loader(cfg_dataset)
    elif dataset_type in ("saocom", "saocom_timeseries"):
          return SAOCOMLoader(cfg_dataset)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_type}")
    
    