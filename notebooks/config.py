import os
import sys
project_root = '/mnt/c/Users/Julien Fournier/Documents'
data_root = '/home/jul/DST/recoRAG'

project_dir = os.path.join(project_root, 'GitHub', 'recoRAG')
if project_dir not in sys.path:
    sys.path.append(project_dir)
    sys.path.append(os.path.join(project_dir, 'src'))

# directory of the project
path_to_project = project_dir
# path to data
path_to_data = os.path.join(data_root, 'data')
# path to trained models
path_to_models = os.path.join(project_dir, 'models')