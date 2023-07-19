import os
import numpy as np
import torch.utils.data as data

from plyfile import PlyData


class ShapeNet(data.Dataset):
    def __init__(self, train='training', options=None):
        self.train = train
        self.use_normals = options.use_normals
        self.rootpc = os.path.join(os.path.abspath('.'), 'data/customShapeNet')
        self.npoints = options.npoint
        self.datapath = []
        self.catfile = os.path.join(os.path.abspath('.'), 'data/synsetoffset2category.txt')
        self.cat = {}
        self.class_choice = [options.class_choice] if options.class_choice is not None else None

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]

        if self.class_choice is not None:
            self.cat = {k: v for k, v in self.cat.items() if k in self.class_choice}
        print(self.cat)

        for item in self.cat:
            dir_point = os.path.join(self.rootpc, self.cat[item], 'ply')
            fns = sorted(os.listdir(dir_point))
            print('category', self.cat[item], 'files ' + str(len(fns)))

            if train == 'training':
                fns = fns[:int(len(fns) * 0.8)]
            elif train == 'validation':
                fns = fns[int(len(fns) * 0.8):]
            elif train == 'full':
                fns = fns

            for fn in fns:
                self.datapath.append(os.path.join(dir_point, fn))

    def __getitem__(self, index):
        fn = self.datapath[index]

        ply_data = PlyData.read(fn)
        points = ply_data['vertex']
        indice = np.array(np.random.choice(points.count, self.npoints, replace=False))
        points = np.vstack([points['x'], points['y'], points['z'], points['nx'], points['ny'], points['nz']]).T
        point_set = points[indice]

        if not self.use_normals:
            point_set = point_set[:, 0:3]

        return point_set

    def __len__(self):
        return len(self.datapath)
