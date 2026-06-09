import torch
from pathlib import Path
pt_dir = Path('D:/pts_aeg/train0a32f9ecaccbc932966bead22f0d0abbaf0e5e14ed02963a9cc049b649a5e3ea.pt')
dims = {}
for pt in pt_dir.rglob('*.pt'):
    data = torch.load(pt, map_location='cpu')
    dim = data['graph'].x.size(1)
    dims[dim] = dims.get(dim, 0) + 1
print('Node_x dimensions:')
for dim, count in sorted(dims.items()):
    print(f'  {dim}: {count} files')
if len(dims) > 1:
    print('❌ Inconsistent dimensions found!')
else:
    print('✅ All PT files have consistent dimensions')