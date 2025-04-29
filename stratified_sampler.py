import numpy as np
from torch.utils.data.sampler import Sampler

class StratifiedFixedSampler(Sampler):
    def __init__(self, energies, batch_size, bins):
        """
        energies: array-like, per-sample energy values
        batch_size: int, number of samples per batch
        bins: list, bin edges for energy ranges, must include min and max values
        """
        self.energies = np.array(energies)
        self.batch_size = batch_size
        
        # Validate bins
        if len(bins) < 2:
            raise ValueError("bins must have at least 2 values (min and max)")
        if not np.all(np.diff(bins) > 0):
            raise ValueError("bins must be strictly increasing")
        if bins[0] > np.min(self.energies):
            raise ValueError(f"First bin edge {bins[0]} is greater than minimum energy {np.min(self.energies)}")
        if bins[-1] < np.max(self.energies):
            raise ValueError(f"Last bin edge {bins[-1]} is less than maximum energy {np.max(self.energies)}")
            
        self.bins = bins

        # 将所有样本归到bin里
        # 使用 right=True 使得边界值进入下一个bin
        self.bin_indices = np.digitize(self.energies, bins=self.bins, right=True)

        # 按bin分组
        self.groups = {}
        for b in np.unique(self.bin_indices):
            self.groups[b] = np.where(self.bin_indices == b)[0]

        # 计算整体比例
        self.bin_counts = np.array([len(self.groups.get(b, [])) for b in range(len(bins))])
        self.total = np.sum(self.bin_counts)
        self.bin_proportions = self.bin_counts / self.total

        self.num_samples = len(self.energies)
        
        # 初始化每个bin的可用样本索引
        self.available_indices = {b: list(self.groups.get(b, [])) for b in range(len(bins))}
        self.used_indices = set()

    def __iter__(self):
        all_indices = []
        
        # 重置可用样本和已用样本
        self.available_indices = {b: list(self.groups.get(b, [])) for b in range(len(self.bins))}
        self.used_indices = set()
        
        # 计算需要的batch数量
        num_batches = int(np.ceil(self.num_samples / self.batch_size))
        
        for _ in range(num_batches):
            batch = []
            # 计算当前batch中每个bin应该采样的数量
            remaining_samples = self.batch_size
            
            # 计算每个bin的剩余样本数
            remaining_in_bins = {b: len(self.available_indices[b]) for b in self.available_indices}
            total_remaining = sum(remaining_in_bins.values())
            
            if total_remaining == 0:
                break
                
            # 根据比例分配每个bin的采样数量
            bin_sample_counts = {}
            for b in self.available_indices:
                if remaining_in_bins[b] > 0:
                    # 根据比例计算应该采样的数量
                    target_count = int(np.ceil(self.bin_proportions[b] * remaining_samples))
                    # 确保不超过剩余样本数
                    bin_sample_counts[b] = min(target_count, remaining_in_bins[b])
                    remaining_samples -= bin_sample_counts[b]
            
            # 如果还有剩余需要分配的样本，按比例分配给还有剩余样本的bin
            if remaining_samples > 0:
                remaining_bins = [b for b in bin_sample_counts if remaining_in_bins[b] > bin_sample_counts[b]]
                while remaining_samples > 0 and remaining_bins:
                    for b in remaining_bins:
                        if remaining_samples > 0 and remaining_in_bins[b] > bin_sample_counts[b]:
                            bin_sample_counts[b] += 1
                            remaining_samples -= 1
                    remaining_bins = [b for b in remaining_bins if remaining_in_bins[b] > bin_sample_counts[b]]
            
            # 从每个bin中采样
            for b, n_samples in bin_sample_counts.items():
                if n_samples > 0 and len(self.available_indices[b]) > 0:
                    # 随机选择n_samples个样本
                    chosen = np.random.choice(self.available_indices[b], size=n_samples, replace=False)
                    # 更新可用样本列表
                    self.available_indices[b] = [idx for idx in self.available_indices[b] if idx not in chosen]
                    batch.extend(chosen)
            
            np.random.shuffle(batch)
            all_indices.extend(batch)

        return iter(all_indices)

    def __len__(self):
        return self.num_samples 