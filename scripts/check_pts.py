import torch
from pathlib import Path

# Fix: This should be a directory path, not a file path
pt_dir = Path('D:/pts_aeg/train')
dims = {}
for pt in pt_dir.rglob('*.pt'):
    data = torch.load(pt, map_location='cpu')
    dim = data['node_x'].size(1)  # Fix: use 'node_x' key from AEG payload
    dims[dim] = dims.get(dim, 0) + 1
print('Node_x dimensions:')
for dim, count in sorted(dims.items()):
    print(f'  {dim}: {count} files')
if len(dims) > 1:
    print('❌ Inconsistent dimensions found!')
else:
    print('✅ All PT files have consistent dimensions')
