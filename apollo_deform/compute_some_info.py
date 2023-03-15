# Calculate mean shape and PCA matrix for reducted obj

save = '../apollo_mean_car_shape'
N = 30

from pytorch3d.io import load_obj, save_obj
import os 
import numpy as np
import torch
import json

with open('id_to_abb.json') as f:
    id_to_abb = json.load(f)

reducted_id = set()
for id, abb in id_to_abb.items():
    reducted_id.add(abb)

# call whole meshes
M = torch.zeros(len(reducted_id),1352,3)
for car in reducted_id:
    vert, face, _ = load_obj(f'{car}.obj')
    M[car] = vert
    
mean = M.mean(0)
# save mean shape
save_obj(os.path.join(save, 'red_mean.obj'), verts=mean, faces=face.verts_idx)

from sklearn.decomposition import PCA

# subtract mean shape
M_hat = M - mean

# compute PCA matrix
pca = PCA(n_components=N)
pca.fit(M_hat.reshape(len(reducted_id),-1))
S = pca.components_.reshape(-1,1352,3)

# save results
np.save(os.path.join(save, 'red_pca.npy'), S)