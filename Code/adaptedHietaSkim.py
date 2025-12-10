BASE_PATH = "/content/drive/MyDrive/TCC"

"""
HieTaSkim para fMRI - FATIAS GLOBAIS + SALVAMENTO POR SUJEITO
==============================================================
Agrega resultados de todos os sujeitos, identifica as N fatias
mais relevantes NO GERAL e salva as fatias de cada sujeito
"""

import numpy as np
import nibabel as nib
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.decomposition import IncrementalPCA
import matplotlib.pyplot as plt
from pathlib import Path
import json
from tqdm import tqdm
import pandas as pd
import warnings
import gc
import psutil
import time

warnings.filterwarnings('ignore')


def get_memory_usage():
    """Retorna uso de memória em GB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024**3


class fMRIHieTaSkim:
    """HieTaSkim para fMRI"""

    def __init__(self, delta_t=5, gamma=0.75, max_summary_ratio=0.15, nc_min=3,
                 spatial_downsample=2, chunk_size=50, use_float32=True):
        self.delta_t = delta_t
        self.gamma = gamma
        self.max_summary_ratio = max_summary_ratio
        self.nc_min = nc_min
        self.spatial_downsample = spatial_downsample
        self.chunk_size = chunk_size
        self.dtype = np.float32 if use_float32 else np.float64

        self.keyshots_ = None
        self.selected_volumes_ = None
        self.slice_scores_ = None
        self.n_volumes_ = None
        self.img_ = None  # 🆕 Guardar referência à imagem

    def load_fmri_data(self, fmri_path):
        """Carrega dados com downsampling espacial"""
        print(f"  └─ Carregando: {Path(fmri_path).name}")
        img = nib.load(fmri_path)
        data_shape = img.shape
        self.n_volumes_ = data_shape[3]
        self.img_ = img  # 🆕 Guardar referência
        print(f"     Shape original: {data_shape}")

        if self.spatial_downsample > 1:
            new_shape = (
                data_shape[0] // self.spatial_downsample,
                data_shape[1] // self.spatial_downsample,
                data_shape[2] // self.spatial_downsample,
                data_shape[3]
            )
            print(f"     Shape após downsample: {new_shape}")
        return img

    def create_mask(self, img, mask_threshold=0.1, sample_volumes=10):
        """Cria máscara cerebral"""
        print(f"  └─ Criando máscara cerebral...")
        n_vols = img.shape[3]
        sample_idx = np.linspace(0, n_vols-1, min(sample_volumes, n_vols), dtype=int)

        mean_img = None
        for idx in sample_idx:
            vol = np.asarray(img.dataobj[:, :, :, idx], dtype=self.dtype)
            if self.spatial_downsample > 1:
                vol = vol[::self.spatial_downsample,
                         ::self.spatial_downsample,
                         ::self.spatial_downsample]
            if mean_img is None:
                mean_img = vol / len(sample_idx)
            else:
                mean_img += vol / len(sample_idx)
            del vol

        threshold = np.percentile(mean_img[mean_img > 0], mask_threshold * 100)
        mask = mean_img > threshold
        print(f"     Voxels na máscara: {np.sum(mask):,}")
        del mean_img
        gc.collect()
        return mask

    def extract_features_chunked(self, img, mask, method='slice_mean'):
        """Extrai features em chunks"""
        print(f"  └─ Extraindo features ({method})...")
        n_volumes = img.shape[3]
        features_list = []
        n_chunks = int(np.ceil(n_volumes / self.chunk_size))

        for chunk_idx in tqdm(range(n_chunks), desc="     Features", leave=False):
            start_vol = chunk_idx * self.chunk_size
            end_vol = min(start_vol + self.chunk_size, n_volumes)

            chunk_data = np.asarray(
                img.dataobj[:, :, :, start_vol:end_vol],
                dtype=self.dtype
            )

            if self.spatial_downsample > 1:
                chunk_data = chunk_data[::self.spatial_downsample,
                                       ::self.spatial_downsample,
                                       ::self.spatial_downsample, :]

            chunk_features = self._extract_slice_mean_features(chunk_data, mask)
            features_list.append(chunk_features)
            del chunk_data
            gc.collect()

        features = np.vstack(features_list)
        del features_list
        gc.collect()
        print(f"     Features shape: {features.shape}")
        return features

    def _extract_slice_mean_features(self, data, mask):
        """Extrai médias por slice"""
        n_vols = data.shape[3]
        features_list = []

        for t in range(n_vols):
            vol = data[:, :, :, t]

            sagital = np.array([
                np.mean(vol[x, :, :][mask[x, :, :]]) if np.any(mask[x, :, :]) else 0
                for x in range(vol.shape[0])
            ])
            coronal = np.array([
                np.mean(vol[:, y, :][mask[:, y, :]]) if np.any(mask[:, y, :]) else 0
                for y in range(vol.shape[1])
            ])
            axial = np.array([
                np.mean(vol[:, :, z][mask[:, :, z]]) if np.any(mask[:, :, z]) else 0
                for z in range(vol.shape[2])
            ])

            features = np.concatenate([sagital, coronal, axial])
            features_list.append(features)

        return np.array(features_list, dtype=self.dtype)

    def build_temporal_graph(self, features, distance_metric='correlation'):
        """Constrói grafo temporal"""
        print(f"  └─ Construindo grafo temporal...")
        n_volumes = features.shape[0]
        adjacency = np.full((n_volumes, n_volumes), np.inf, dtype=self.dtype)

        for i in tqdm(range(n_volumes), desc="     Grafo", leave=False):
            j_start = max(0, i - self.delta_t)
            j_end = min(n_volumes, i + self.delta_t + 1)

            for j in range(j_start, j_end):
                if i != j:
                    if distance_metric == 'correlation':
                        corr = np.corrcoef(features[i], features[j])[0, 1]
                        dist = 1 - corr if not np.isnan(corr) else 1.0
                    else:
                        dist = np.linalg.norm(features[i] - features[j])
                    adjacency[i, j] = dist

        return adjacency

    def compute_mst(self, adjacency):
        """Calcula MST"""
        print(f"  └─ Calculando MST...")
        adjacency_sparse = csr_matrix(adjacency)
        mst = minimum_spanning_tree(adjacency_sparse)
        del adjacency_sparse
        gc.collect()
        return mst

    def adaptive_hierarchical_cut(self, mst):
        """Cortes hierárquicos adaptativos"""
        print(f"  └─ Realizando cortes hierárquicos...")
        mst_dense = mst.toarray()

        edges = []
        for i in range(mst_dense.shape[0]):
            for j in range(i+1, mst_dense.shape[1]):
                if mst_dense[i, j] > 0:
                    edges.append((i, j, mst_dense[i, j]))

        edges.sort(key=lambda x: x[2], reverse=True)
        parent = list(range(mst_dense.shape[0]))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
                return True
            return False

        def equilibrium_measure(component_edges):
            if len(component_edges) == 0:
                return 0
            weights = [e[2] for e in component_edges]
            return np.mean(weights) + self.gamma * np.std(weights)

        cuts_made = 0
        for edge in edges:
            i, j, weight = edge
            if find(i) == find(j):
                continue

            component = {node for node in range(len(parent)) if find(node) == find(i)}
            component_edges = [e for e in edges if find(e[0]) in component or find(e[1]) in component]
            F_e = equilibrium_measure(component_edges)

            if weight >= F_e or cuts_made < self.nc_min - 1:
                union(i, j)
                cuts_made += 1

        components = {}
        for node in range(len(parent)):
            root = find(node)
            if root not in components:
                components[root] = []
            components[root].append(node)

        keyshots = [sorted(comp) for comp in components.values()]
        keyshots.sort(key=lambda x: x[0])
        if len(keyshots) < 1:
            keyshots = [list(range(0, mst_dense.shape[0], 2))]
        print(f"     {len(keyshots)} keyshots identificados")
        return keyshots

    def select_representative_volumes(self, keyshots):
        """Seleciona volumes representativos"""
        max_frames = int(self.n_volumes_ * self.max_summary_ratio)
        frames_per_keyshot = max(1, max_frames // len(keyshots))

        selected = []
        for ks in keyshots:
            if not ks:
                continue

            center_idx = ks[len(ks) // 2]
            n_frames = min(frames_per_keyshot, len(ks))
            half = n_frames // 2
            start_idx = max(0, center_idx - half)
            end_idx = min(center_idx + half + 1, self.n_volumes_)

            for idx in range(start_idx, end_idx):
                selected.append(idx)

        selected = sorted(set(selected))

        if len(selected) == 0:
            print("⚠️ Nenhum volume selecionado! Usando volume central como fallback.")
            selected = [self.n_volumes_ // 2]

        print(f"  └─ {len(selected)}/{self.n_volumes_} volumes ({100*len(selected)/self.n_volumes_:.1f}%)")
        return selected

    def compute_slice_scores(self, img, mask, selected_volumes):
        """Calcula scores para TODAS as fatias"""
        print(f"  └─ Calculando scores de todas as fatias...")

        if not selected_volumes or len(selected_volumes) == 0:
            print("     ⚠️  Nenhum volume selecionado! Retornando scores vazios.")
            return {
                'sagital': np.array([]),
                'coronal': np.array([]),
                'axial': np.array([])
            }

        mean_activation = None
        n_processed = 0

        for i in range(0, len(selected_volumes), self.chunk_size):
            chunk_vols = selected_volumes[i:i+self.chunk_size]
            chunk_data = []

            for vol_idx in chunk_vols:
                try:
                    vol = np.asarray(img.dataobj[:, :, :, vol_idx], dtype=self.dtype)
                    if self.spatial_downsample > 1:
                        vol = vol[::self.spatial_downsample,
                                 ::self.spatial_downsample,
                                 ::self.spatial_downsample]
                    chunk_data.append(vol)
                except Exception as e:
                    print(f"     ⚠️  Erro ao carregar volume {vol_idx}: {e}")
                    continue

            if not chunk_data:
                continue

            chunk_array = np.stack(chunk_data, axis=3)
            chunk_mean = np.mean(chunk_array, axis=3)

            if mean_activation is None:
                mean_activation = chunk_mean
            else:
                mean_activation += chunk_mean

            n_processed += len(chunk_data)

            del chunk_data, chunk_array, chunk_mean
            gc.collect()

        if mean_activation is None or n_processed == 0:
            print("     ⚠️  Nenhum dado processado! Retornando scores vazios.")
            return {
                'sagital': np.array([]),
                'coronal': np.array([]),
                'axial': np.array([])
            }

        print(f"     Processados {n_processed}/{len(selected_volumes)} volumes")
        mean_activation /= n_processed
        mean_activation_masked = mean_activation * mask

        slice_scores = {
            'sagital': [],
            'coronal': [],
            'axial': []
        }

        # SAGITAL
        for x in range(mean_activation.shape[0]):
            if np.any(mask[x, :, :]):
                variance = np.var(mean_activation_masked[x, :, :][mask[x, :, :]])
                n_voxels = np.sum(mask[x, :, :])
                score = variance * n_voxels
            else:
                score = 0
            slice_scores['sagital'].append(score)

        # CORONAL
        for y in range(mean_activation.shape[1]):
            if np.any(mask[:, y, :]):
                variance = np.var(mean_activation_masked[:, y, :][mask[:, y, :]])
                n_voxels = np.sum(mask[:, y, :])
                score = variance * n_voxels
            else:
                score = 0
            slice_scores['coronal'].append(score)

        # AXIAL
        for z in range(mean_activation.shape[2]):
            if np.any(mask[:, :, z]):
                variance = np.var(mean_activation_masked[:, :, z][mask[:, :, z]])
                n_voxels = np.sum(mask[:, :, z])
                score = variance * n_voxels
            else:
                score = 0
            slice_scores['axial'].append(score)

        for axis in slice_scores:
            slice_scores[axis] = np.array(slice_scores[axis])

        del mean_activation, mean_activation_masked
        gc.collect()

        return slice_scores

    def fit(self, fmri_path, feature_method='slice_mean', distance_metric='correlation'):
        """Executa pipeline completo"""
        print(f"\n{'='*70}")
        print(f"Processando: {Path(fmri_path).name}")
        print(f"{'='*70}")

        img = self.load_fmri_data(fmri_path)
        mask = self.create_mask(img)
        features = self.extract_features_chunked(img, mask, method=feature_method)
        adjacency = self.build_temporal_graph(features, distance_metric=distance_metric)
        mst = self.compute_mst(adjacency)

        del adjacency, features
        gc.collect()

        keyshots = self.adaptive_hierarchical_cut(mst)
        self.keyshots_ = keyshots

        selected = self.select_representative_volumes(keyshots)
        self.selected_volumes_ = selected

        slice_scores = self.compute_slice_scores(img, mask, selected)
        self.slice_scores_ = slice_scores

        print(f"\n✓ Concluído!")
        print(f"{'='*70}\n")

        return self


class BatchfMRIProcessor:
    """Processa batch e AGREGA scores globais + salva fatias por sujeito"""

    def __init__(self, dataset_path, task_name='stopsignal', **hietaskim_params):
        self.dataset_path = Path(dataset_path)
        self.task_name = task_name
        self.hietaskim_params = hietaskim_params
        self.results = {}
        self.global_slice_scores = None
        self.subject_models = {}  # 🆕 Guardar modelos por sujeito

    def find_fmri_files(self):
        """Encontra arquivos fMRI"""
        print("\n" + "="*70)
        print("Procurando arquivos fMRI...")
        print("="*70)
        print(f"📁 Diretório base: {self.dataset_path}")
        print(f"🔍 Padrão de busca: *_task-{self.task_name}_bold.nii.gz")
        print()

        if not self.dataset_path.exists():
            print(f"❌ ERRO: Diretório não existe!")
            print(f"   Caminho: {self.dataset_path}")
            return []

        subdirs = list(self.dataset_path.glob("sub-*"))
        print(f"📂 Encontrados {len(subdirs)} subdiretórios 'sub-*'")

        if len(subdirs) == 0:
            print("\n⚠️  Nenhum subdiretório 'sub-*' encontrado!")
            print("   Estrutura esperada:")
            print("   dataset/")
            print("   ├── sub-01/")
            print("   ├── sub-02/")
            print("   └── ...")

            print(f"\n   Conteúdo de {self.dataset_path}:")
            for item in list(self.dataset_path.iterdir())[:10]:
                print(f"     - {item.name}")
            return []

        files = []
        pattern = f"*_task-{self.task_name}_bold.nii.gz"

        for sub_dir in sorted(subdirs):
            if sub_dir.is_dir():
                fmri_files = list(sub_dir.rglob(pattern))

                if fmri_files:
                    subject_id = sub_dir.name
                    files.append((subject_id, fmri_files[0]))
                    print(f"  ✓ {subject_id}: {fmri_files[0].name}")
                else:
                    print(f"  ✗ {sub_dir.name}: nenhum arquivo '{pattern}' encontrado")
                    all_files = list(sub_dir.rglob("*.nii.gz"))
                    if all_files:
                        print(f"      Mas encontrados: {[f.name for f in all_files[:3]]}")

        print(f"\n{'✓' if files else '❌'} {len(files)} sujeitos com arquivos válidos")
        print("="*70)
        return files

    def process_all_subjects(self, feature_method='slice_mean', distance_metric='correlation'):
        """Processa todos e ACUMULA scores"""
        files = self.find_fmri_files()

        if not files:
            print("❌ Nenhum arquivo encontrado!")
            return {}

        print(f"\n{'='*70}")
        print(f"Processando {len(files)} sujeitos...")
        print(f"{'='*70}")

        accumulated_scores = None
        n_subjects = 0

        for subject_id, fmri_file in files:
            try:
                model = fMRIHieTaSkim(**self.hietaskim_params)
                model.fit(str(fmri_file),
                         feature_method=feature_method,
                         distance_metric=distance_metric)

                # 🆕 Guardar modelo
                self.subject_models[subject_id] = model

                if accumulated_scores is None:
                    accumulated_scores = {
                        'sagital': model.slice_scores_['sagital'].copy(),
                        'coronal': model.slice_scores_['coronal'].copy(),
                        'axial': model.slice_scores_['axial'].copy()
                    }
                else:
                    for axis in ['sagital', 'coronal', 'axial']:
                        min_len = min(len(accumulated_scores[axis]),
                                     len(model.slice_scores_[axis]))
                        accumulated_scores[axis][:min_len] += model.slice_scores_[axis][:min_len]

                n_subjects += 1

                self.results[subject_id] = {
                    'n_volumes': model.n_volumes_,
                    'n_selected': len(model.selected_volumes_),
                    'n_keyshots': len(model.keyshots_)
                }

                print(f"✓ {subject_id} processado ({n_subjects}/{len(files)})")
                print(f"  RAM: {get_memory_usage():.2f} GB\n")

            except Exception as e:
                print(f"\n❌ Erro em {subject_id}: {str(e)}\n")
                continue

        if accumulated_scores and n_subjects > 0:
            for axis in accumulated_scores:
                accumulated_scores[axis] /= n_subjects

        self.global_slice_scores = accumulated_scores
        print(f"\n✓ {n_subjects} sujeitos processados com sucesso!")
        return self.results

    def get_top_slices(self, n_slices_per_axis=5):
        """Retorna as TOP N fatias globais"""
        if self.global_slice_scores is None:
            raise ValueError("Execute process_all_subjects() primeiro!")

        top_slices = {}

        for axis, scores in self.global_slice_scores.items():
            sorted_indices = np.argsort(scores)[::-1]
            top_n = sorted_indices[:n_slices_per_axis]

            top_slices[axis] = [
                {
                    'rank': rank + 1,
                    'index': int(idx),
                    'score': float(scores[idx])
                }
                for rank, idx in enumerate(top_n)
            ]

        return top_slices


    def save_subject_slices(self, subject_id, top_slices, output_dir, volume_index=0):
        """
        🆕 Salva as top N fatias de um sujeito específico

        Args:
            subject_id: ID do sujeito
            top_slices: Dict com as fatias globais top N
            output_dir: Diretório de saída
            volume_index: Índice do volume temporal a extrair (padrão: 0)
        """
        if subject_id not in self.subject_models:
            print(f"⚠️  Modelo não encontrado para {subject_id}")
            return

        model = self.subject_models[subject_id]
        img = model.img_

        # Criar pasta do sujeito
        subject_dir = Path(output_dir) / subject_id
        subject_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  💾 Salvando fatias de {subject_id}...")

        # Carregar volume
        vol = np.asarray(img.dataobj[:, :, :, volume_index], dtype=np.float32)

        # Salvar cada tipo de fatia
        for axis in ['sagital', 'coronal', 'axial']:
            axis_dir = subject_dir / axis
            axis_dir.mkdir(exist_ok=True)

            for slice_info in top_slices[axis]:
                idx = slice_info['index']
                rank = slice_info['rank']

                # Extrair fatia
                if axis == 'sagital':
                    slice_data = vol[idx, :, :]
                elif axis == 'coronal':
                    slice_data = vol[:, idx, :]
                else:  # axial
                    slice_data = vol[:, :, idx]

                # Salvar como .npy
                filename = f"slice_{axis}_{rank:02d}_idx{idx:03d}.npy"
                np.save(axis_dir / filename, slice_data)

                # Salvar como imagem PNG (apenas a imagem, sem anotações)
                plt.figure(figsize=(8, 8))
                plt.imshow(slice_data.T, cmap='gray', origin='lower')
                plt.axis('off')  # Remove eixos
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove margens

                png_filename = f"slice_{axis}_{rank:02d}_idx{idx:03d}.png"
                plt.savefig(axis_dir / png_filename, dpi=100, bbox_inches='tight', pad_inches=0)
                plt.close()

        print(f"     ✓ Fatias salvas em: {subject_dir}")

    def save_results(self, output_dir, n_slices_per_axis=5, save_subject_slices=True):
        """
        🆕 Salva resultados com fatias globais E fatias individuais por sujeito

        Args:
            output_dir: Diretório de saída
            n_slices_per_axis: Número de fatias a salvar por eixo
            save_subject_slices: Se True, salva fatias de cada sujeito
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*70}")
        print("Salvando resultados...")
        print(f"{'='*70}")

        # Pegar top slices globais
        top_slices = self.get_top_slices(n_slices_per_axis)

        # Salvar fatias globais (como antes)
        with open(output_path / 'global_best_slices.txt', 'w') as f:
            f.write("="*70 + "\n")
            f.write(f"FATIAS MAIS RELEVANTES NO GERAL (Top {n_slices_per_axis})\n")
            f.write(f"Agregado de {len(self.results)} sujeitos\n")
            f.write("="*70 + "\n\n")

            for axis in ['sagital', 'coronal', 'axial']:
                f.write(f"{axis.upper()}:\n")
                for slice_info in top_slices[axis]:
                    f.write(f"  #{slice_info['rank']}: Índice {slice_info['index']:3d} | "
                           f"Score médio: {slice_info['score']:.6e}\n")
                f.write("\n")

        with open(output_path / 'global_best_slices.json', 'w') as f:
            json.dump({
                'n_subjects': len(self.results),
                'n_slices_per_axis': n_slices_per_axis,
                'top_slices': top_slices
            }, f, indent=2)

        for axis in self.global_slice_scores:
            np.save(output_path / f'global_scores_{axis}.npy',
                   self.global_slice_scores[axis])

        df = pd.DataFrame([
            {
                'subject_id': sid,
                'n_volumes': res['n_volumes'],
                'n_selected': res['n_selected'],
                'n_keyshots': res['n_keyshots']
            }
            for sid, res in self.results.items()
        ])
        df.to_csv(output_path / 'summary_subjects.csv', index=False)

        # 🆕 Salvar fatias individuais de cada sujeito
        if save_subject_slices:
            print(f"\n{'='*70}")
            print(f"Salvando fatias individuais de {len(self.subject_models)} sujeitos...")
            print(f"{'='*70}")

            for subject_id in tqdm(self.subject_models.keys(), desc="Salvando fatias"):
                try:
                    self.save_subject_slices(subject_id, top_slices, output_path)
                except Exception as e:
                    print(f"\n❌ Erro ao salvar fatias de {subject_id}: {e}")
                    continue

        print("\n✓ Arquivos salvos:")
        print(f"  📄 global_best_slices.txt")
        print(f"  📄 global_best_slices.json")
        print(f"  📄 global_scores_*.npy")
        print(f"  📄 summary_subjects.csv")
        if save_subject_slices:
            print(f"  📁 {len(self.subject_models)} pastas de sujeitos com fatias")
        print(f"\n✓ Resultados em: {output_path}")
        print("="*70)


if __name__ == "__main__":

    print("\n" + "="*70)
    print("HieTaSkim - Extração de Fatias Globais + Individuais")
    print("="*70)

    # PASSO 1: Configurar processador
    processor = BatchfMRIProcessor(
        dataset_path='/content/drive/MyDrive/TCC/dataset/ds000030-download',
        task_name='stopsignal',
        delta_t=5,
        gamma=0.75,
        max_summary_ratio=0.15,
        spatial_downsample=0,
        chunk_size=50
    )

    # PASSO 2: Processar TODOS os sujeitos
    print("\n🔄 Iniciando processamento de todos os sujeitos...")
    print("   (Isso pode demorar...)\n")

    results = processor.process_all_subjects(
        feature_method='slice_mean',
        distance_metric='correlation'
    )

    if not results:
        print("\n❌ Nenhum sujeito foi processado!")
        exit()

    # PASSO 3: Extrair TOP N fatias globais
    print("\n🎯 Extraindo fatias mais relevantes no geral...")

    top_5_global = processor.get_top_slices(n_slices_per_axis=16)

    print("\n" + "="*70)
    print("🏆 TOP 5 FATIAS MAIS RELEVANTES NO GERAL:")
    print(f"   (Agregado de {len(results)} sujeitos)")
    print("="*70)

    for axis in ['sagital', 'coronal', 'axial']:
        print(f"\n{axis.upper()}:")
        for s in top_5_global[axis]:
            print(f"  #{s['rank']}: Índice {s['index']:3d} (score: {s['score']:.6e})")

    # PASSO 4: Salvar resultados (com fatias individuais)
    print("\n" + "="*70)
    print("💾 Salvando resultados globais e fatias individuais...")
    print("="*70)

    processor.save_results(
        output_dir='/content/drive/MyDrive/TCC/resultados_global',
        n_slices_per_axis=16,
        save_subject_slices=True  # 🆕 Ativa salvamento de fatias por sujeito
    )

    print("\n" + "="*70)
    print("✅ Processamento completo!")
    print("="*70)

    # BÔNUS: Estrutura de arquivos gerada
    print("\n📂 Estrutura de arquivos gerada:")
    print("   resultados_global/")
    print("   ├── global_best_slices.txt")
    print("   ├── global_best_slices.json")
    print("   ├── global_scores_sagital.npy")
    print("   ├── global_scores_coronal.npy")
    print("   ├── global_scores_axial.npy")
    print("   ├── summary_subjects.csv")
    print("   ├── sub-01/")
    print("   │   ├── sagital/")
    print("   │   │   ├── slice_sagital_01_idxXXX.npy")
    print("   │   │   ├── slice_sagital_01_idxXXX.png")
    print("   │   │   └── ...")
    print("   │   ├── coronal/")
    print("   │   │   └── ...")
    print("   │   └── axial/")
    print("   │       └── ...")
    print("   ├── sub-02/")
    print("   │   └── ...")
    print("   └── ...")

    # BÔNUS 2: Como usar os resultados
    print("\n📖 Como usar os resultados:")
    print("   1. As fatias globais estão em: top_5_global")
    print("   2. Exemplo de acesso:")
    print(f"      - Melhor fatia sagital: índice {top_5_global['sagital'][0]['index']}")
    print(f"      - Melhor fatia coronal: índice {top_5_global['coronal'][0]['index']}")
    print(f"      - Melhor fatia axial: índice {top_5_global['axial'][0]['index']}")
    print("\n   3. As fatias de cada sujeito já estão salvas em:")
    print("      - Formato .npy (arrays numpy)")
    print("      - Formato .png (imagens)")
    print("\n   4. Para carregar uma fatia específica:")
    print("      slice_data = np.load('resultados_global/sub-01/axial/slice_axial_01_idxXXX.npy')")
    print("\n   5. Para processar apenas alguns sujeitos:")
    print("      # Modifique save_subject_slices para processar sujeitos específicos")

    print("\n" + "="*70)

