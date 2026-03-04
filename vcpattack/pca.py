import torch
from sklearn.decomposition import PCA

class AdaptivePCASpace:
    def __init__(self, subspace_dim=50):
        self.subspace_dim = subspace_dim
        self.U_k = None

    def update_subspace(self, feats):
        feats_np = feats.detach().cpu().numpy()
        pca = PCA(n_components=self.subspace_dim)
        pca.fit(feats_np)
        self.U_k = torch.tensor(pca.components_.T, dtype=torch.float32).to(feats.device)

    def project(self, feats):
        return feats @ self.U_k

    def inverse_project_grad(self, grad_proj):
        return grad_proj @ self.U_k.T
