import torch
import torch.nn as nn
from einops import rearrange
from tqdm import tqdm
from sklearn.cluster import KMeans
from .metric import *
from utils.coordconv import CoordConv2d
import torch.nn.functional as F
from torch.nn.functional import pairwise_distance

class DSVDD(nn.Module):
    def __init__(self, model, data_loader, cnn, gamma_c, gamma_d, sigmoid, memory_update, loss, positives, negatives, num_prototypes, add_coord, avg_pooling, device):
        super(DSVDD, self).__init__()
        self.device = device
        
        self.num_prototypes = num_prototypes
        self.C   = 0
        self.nu = 1e-3
        self.scale = 56
        self.memory_update = memory_update
        self.avg_pooling = avg_pooling
        self.gamma_c = gamma_c
        self.gamma_d = gamma_d
        self.alpha = 1e-1
        self.positives = positives
        self.negatives = negatives
        self.loss = loss
        self.r   = nn.Parameter(1e-5*torch.ones(1), requires_grad=True)
        self.Descriptor = Descriptor(self.gamma_d, add_coord, cnn, sigmoid, avg_pooling).to(device)
        self._init_centroid(model, data_loader)
        #self.C = rearrange(self.C, 'b c h w -> (b h w) c').detach()
        """
        if self.gamma_c > 1:
            self.C = self.C.cpu().detach().numpy()
            self.C = KMeans(n_clusters=(self.scale**2)//self.gamma_c, max_iter=3000).fit(self.C).cluster_centers_
            self.C = torch.Tensor(self.C).to(device)
        """
        #self.C = self.C.transpose(-1, -2).detach()
        
        self.C = nn.Parameter(self.C, requires_grad= memory_update == "memory_parameter_update")
        
        
            

    def forward(self, p):
        phi_p = self.Descriptor(p)
        phi_p = rearrange(phi_p, 'b c h w -> b (h w) c')

        features = torch.sum(torch.pow(phi_p, 2), 2, keepdim=True)    
        centers  = torch.sum(torch.pow(self.C, 2), 0, keepdim=True)
        f_c      = 2 * torch.matmul(phi_p, (self.C))
        dist     = features + centers - f_c
        dist     = torch.sqrt(dist)

        n_neighbors = self.positives
        dist     = dist.topk(n_neighbors, largest=False).values

        dist = (F.softmin(dist, dim=-1)[:, :, 0]) * dist[:, :, 0]
        dist = dist.unsqueeze(-1)
        score = rearrange(dist, 'b (h w) c -> b c h w', h=self.scale)
        
        loss = 0
        if self.training:
            if self.loss == "online":
                loss = self._online_loss(phi_p)
            if self.loss == "cfa":
                loss = self._soft_boundary(phi_p)
            if self.loss == "tripletcentral":
                loss = self._triplet_central_loss(phi_p)
            if self.loss == "online_nologexp":
                loss = self._online_loss_nologexp(phi_p)
            if self.loss == "NCE":
                loss = self._NCE_loss(phi_p)
            if self.loss == "email":
                loss = self._email_loss(phi_p)
            
        return loss, score, phi_p

    def _soft_boundary(self, phi_p):

        features = torch.sum(torch.pow(phi_p, 2), 2, keepdim=True)
        centers  = torch.sum(torch.pow(self.C, 2), 0, keepdim=True)
        f_c      = 2 * torch.matmul(phi_p, (self.C))
        dist     = features + centers - f_c
        n_neighbors = self.positives + self.negatives
        
        dist     = dist.topk(n_neighbors, largest=False).values

        score = (dist[:, : , :self.positives] - self.r**2) 
        L_att = (1/self.nu) * torch.mean(torch.max(torch.zeros_like(score), score))
        
        score = (self.r**2 - dist[:, : , self.positives:]) 
        L_rep  = (1/self.nu) * torch.mean(torch.max(torch.zeros_like(score), score - self.alpha))
        
        loss = L_att + L_rep

        return loss 
    
    def _email_loss(self, features):
        memory_bank = self.C.transpose(0, 1)
        num_prototypes = memory_bank.shape[0]
        
        distances = torch.cdist(features, memory_bank, p=2)
        
        n_neighbors = self.positives + self.negatives
        
        d = distances.topk(n_neighbors, largest=False).values.squeeze()
        
        #d = torch.exp(-d)
        #d = torch.softmax(d, dim=1)
        
        loss=torch.relu(d[:,:self.positives] - self.r) + torch.relu(self.r - d[:,self.positives:] + d[:,:self.positives])
        return loss.mean()

    def _online_loss(self, features):
        memory_bank = self.C.transpose(0, 1)
        num_prototypes = memory_bank.shape[0]
        
        distances = torch.cdist(features, memory_bank, p=2) - self.r  #Calculate pairwise distances
        
        d = torch.max(distances, torch.zeros_like(distances).to(self.device))
        
        n_neighbors = self.positives + self.negatives
        
        d = d.topk(n_neighbors, largest=False).values.squeeze()
        
        d = torch.exp(-d)
        
        min_sum = torch.sum(d[:,:self.positives], dim=1, keepdim=True)
        max_sum = torch.sum(d[:,self.positives:], dim=1, keepdim=True)
        
        el = min_sum / (min_sum + max_sum)
        el = - torch.log(el)
        return el.mean()
    
    def _NCE_loss(self, features):
        memory_bank = self.C.transpose(0, 1)
        num_prototypes = memory_bank.shape[0]
        
        distances = torch.cdist(features, memory_bank, p=2) - self.r  #Calculate pairwise distances
        
        d = torch.max(distances, torch.zeros_like(distances).to(self.device))

        n_neighbors = self.positives + self.negatives
        
        d = d.topk(n_neighbors, largest=False).values.squeeze()
        
        d = torch.sigmoid(d)

        #d[:,self.positives:] = 1 - d[:,self.positives:]
        d_clone = d.clone()

        # Perform the operation on the copy
        d_clone[:, self.positives:] = 1 - d_clone[:, self.positives:]
        
        log  = torch.log(d_clone)
        
        summ = - torch.sum(log, dim=1, keepdim=True)

        return summ.mean()
    
    def _NCE_softmax_loss(self, features):
        memory_bank = self.C.transpose(0, 1)
        num_prototypes = memory_bank.shape[0]
        
        distances = torch.cdist(features, memory_bank, p=2) - self.r  #Calculate pairwise distances
        
        d = torch.max(distances, torch.zeros_like(distances).to(self.device))

        n_neighbors = self.positives + self.negatives
        
        d = d.topk(self.positives, largest=False).values.squeeze()
        
        d = torch.softmax(d, dim=1)

        #d[:,self.positives:] = 1 - d[:,self.positives:]
        d_clone = d.clone()

        # Perform the operation on the copy
        d_clone[:, self.positives:] = 1 - d_clone[:, self.positives:]
        
        log  = torch.log(d_clone)
        
        summ = - torch.sum(log, dim=1, keepdim=True)

        return summ.mean()
    
    def _online_loss_nologexp(self, features):
        memory_bank = self.C.transpose(0, 1)
        num_prototypes = memory_bank.shape[0]
        
        distances = torch.cdist(features, memory_bank, p=2) - self.r  #Calculate pairwise distances
        
        d = torch.max(distances, torch.zeros_like(distances).to(self.device))
        
        n_neighbors = self.positives + self.negatives
        
        d = d.topk(n_neighbors, largest=False).values.squeeze()
        
        #d = torch.exp(-d)
        
        min_sum = torch.sum(d[:,:self.positives], dim=1, keepdim=True)
        max_sum = torch.sum(d[:,self.positives:], dim=1, keepdim=True)
        
        el = min_sum / (min_sum + max_sum)
        #el = - torch.log(el)
        return el.mean()
    
    def _triplet_central_loss(self, phi_p):
        
        features = torch.sum(torch.pow(phi_p, 2), 2, keepdim=True)
        centers  = torch.sum(torch.pow(self.C, 2), 0, keepdim=True)
        f_c      = 2 * torch.matmul(phi_p, (self.C))
        dist     = features + centers - f_c
        n_neighbors = self.positives + self.negatives
        
        dist     = dist.topk(n_neighbors, largest=False).values
        
        L_att = torch.mean(dist[:, : , :self.positives])
        
        L_rep  = torch.mean(dist[:, : , self.positives:])
        
        closest_distance = torch.mean(dist[:, : , 0])
        
        triplet = L_att - L_rep
        
        triplet = torch.max(triplet,torch.zeros_like(triplet))
        
        loss = triplet + closest_distance

        return loss
    
    def _angular_loss(self, features):
        feat_norm = F.normalize(features.squeeze(), p=2, dim=1)
        prototype_norm = F.normalize(self.C, p=2, dim=0)
        
        pairwise_cosine_distance = 1 - torch.mm(feat_norm, prototype_norm)
        n_neighbors = self.positives + self.negatives
        
        dist     = pairwise_cosine_distance.topk(n_neighbors, largest=False).values

        d = torch.max(dist, torch.zeros_like(dist).to(self.device))
        
        n_neighbors = self.positives + self.negatives
        
        d = d.topk(n_neighbors, largest=False).values.squeeze()
        
        #d = torch.exp(-d)
        
        min_sum = torch.sum(d[:,:self.positives], dim=1, keepdim=True)
        max_sum = torch.sum(d[:,self.positives:], dim=1, keepdim=True)
        
        el = min_sum / (min_sum + max_sum)
        #el = - torch.log(el)
        return el.mean()
    
    def _init_centroid(self, model, data_loader):
        
        random_noise = np.random.standard_normal(size=(1792, self.num_prototypes))
        Q, _ = np.linalg.qr(random_noise)
        #Q = np.interp(Q, (Q.min(), Q.max()), (0, 1))
        
        self.C= torch.from_numpy(Q).to(dtype=torch.float)#.permute(1, 0)
        """
        for i, (x, _, _) in enumerate(tqdm(data_loader)):
            x = x.to(self.device)
            p = model(x)
            self.scale = p[0].size(2)
            phi_p = self.Descriptor(p)
            self.C = ((self.C * i) + torch.mean(phi_p, dim=0, keepdim=True).detach()) / (i+1)
        """
    def _get_freq(self, phi_p, memory_bank):
        
        closest_indices = self.assign(phi_p, memory_bank)
        unique_indices, counts = closest_indices.unique(return_counts=True)
        zero_freq = set(range(self.num_prototypes)) - set(unique_indices.cpu().numpy())
        return unique_indices, counts, zero_freq, closest_indices
    
    def _mem_bank_withoutindex(self, memory_bank, index):
        indeces = torch.tensor([x for x in range(self.num_prototypes)]).cuda()
        indeces = indeces[indeces != index]
        new_mememory_bank = torch.index_select(memory_bank, 0, indeces)
        return new_mememory_bank
    
    def _interation_with_index(self, index, phi_p):
        new_mememory_bank = self._mem_bank_withoutindex(self.C.transpose(0, 1),index)
        unique_indices, counts, _, closest_indices= self._get_freq(phi_p, new_mememory_bank)
        max_index = unique_indices[torch.argmax(counts)]
        mask= torch.eq(closest_indices.unsqueeze(1), max_index)
        new_phi_p= phi_p.squeeze()[mask.squeeze()]
        kmeans = KMeans(n_clusters=2, n_init="auto")
        kmeans.fit(new_phi_p.detach().cpu().numpy())

        cluster_centers = kmeans.cluster_centers_

        self.C[:, index] = torch.tensor(cluster_centers[0], requires_grad = False).reshape(-1, 1).squeeze().cuda()
        self.C[:, max_index + 1 if index <= max_index else max_index] = torch.tensor(cluster_centers[1], requires_grad = False).reshape(-1, 1).reshape(-1,1).squeeze().cuda()


    def kmeans_update(self, phi_p):
        threshold = (phi_p.shape[1] / self.num_prototypes) * 0.5 
        memory_bank = self.C.transpose(0, 1)
        unique_indices, counts, zero_freq, _= self._get_freq(phi_p, memory_bank)
        count = 0
        max_interations = self.num_prototypes * 2
        while min(counts)<threshold or zero_freq:
            count+= 1
            if count > max_interations:
                break
            for i in zero_freq:
                self._interation_with_index(i,phi_p)
                unique_indices, counts, zero_freq, _= self._get_freq(phi_p, memory_bank)
                
            for (index, freq) in zip(unique_indices, counts):
                if freq<threshold:
                    self._interation_with_index(index,phi_p)
                    
                    unique_indices, counts, zero_freq, _= self._get_freq(phi_p, memory_bank)
                    break
    def assign(self, features, memory_bank):
        num_prototypes = memory_bank.shape[0]
        
        distances = torch.cdist(features.squeeze(), memory_bank, p=2)
        
        closest_indices = torch.argmin(distances, dim=1, keepdim=False)

        return closest_indices
        
        
class Descriptor(nn.Module):
    def __init__(self, gamma_d, add_coord, cnn, sigmoid, avg_pooling):
        super(Descriptor, self).__init__()
        self.cnn = cnn
        if cnn == 'wrn50_2':
            dim = 1792 
            self.layer = CoordConv2d(dim, dim//gamma_d, add_coord, 1)
        elif cnn == 'res18':
            dim = 448
            self.layer = CoordConv2d(dim, dim//gamma_d, add_coord, 1)
        elif cnn == 'effnet-b5':
            dim = 568
            self.layer = CoordConv2d(dim, 2*dim//gamma_d, add_coord, 1)
        elif cnn == 'vgg19':
            dim = 1280 
            self.layer = CoordConv2d(dim, dim//gamma_d, add_coord, 1)
        
        self.sigmoid_backbone = sigmoid
        self.avg_pooling = avg_pooling
        self.sigmoid = nn.Sigmoid()

    def forward(self, p):
        sample = None
        if self.avg_pooling:
            for o in p:
                o = F.avg_pool2d(o, 3, 1, 1) / o.size(1) if self.cnn == 'effnet-b5' else F.avg_pool2d(o, 3, 1, 1)
                sample = o if sample is None else torch.cat((sample, F.interpolate(o, sample.size(2), mode='bilinear')), dim=1)
        
        else:
            for o in p:
                sample = o if sample is None else torch.cat((sample, F.interpolate(o, sample.size(2), mode='bilinear')), dim=1)
        
        if self.sigmoid_backbone:
            sample = self.sigmoid(sample)
            
        phi_p = self.layer(sample)
            
        return phi_p
