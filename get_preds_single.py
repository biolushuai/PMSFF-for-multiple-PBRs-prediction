import os
import csv
import torch
import pickle
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model_path = r'./models_saved/models_saved_best_t5xluf_clean/model1_pr0.5127.tar'

test_protein_path = r'../test_data/A0QY29_preds.csv'
test_protein = np.loadtxt(test_protein_path, delimiter=",", dtype=float)

save_path = r'./test_data'


from models import A3C3GRUModel
model = A3C3GRUModel()

model_sd = torch.load(model_path)
model.load_state_dict(model_sd)
if torch.cuda.is_available():
    model = model.cuda()

model.eval()


if torch.cuda.is_available():
    test_pbs_vertex = torch.FloatTensor(test_protein).cuda()
else:
    test_pbs_vertex = torch.FloatTensor(test_protein)

p_preds = model(test_pbs_vertex)
p_preds = p_preds.data.cpu().numpy()

protein_result = os.path.join(save_path, 'preds.csv')
with open(protein_result, 'w') as f:
    csv_writer = csv.writer(f, delimiter=',')
    csv_writer.writerow(p_preds)
