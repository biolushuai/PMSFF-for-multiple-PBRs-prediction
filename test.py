import os
import csv
import torch
import pickle
from metrics import *


def test(model, test_proteins):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    all_labels = []
    all_preds = []

    all_auc_roc = []
    all_auc_pr = []

    all_acc = []
    all_mcc = []
    all_f_score = []
    all_precision = []
    all_recall = []
    all_sc = []
    all_sp = []

    for test_p in test_proteins:
        # print(test_g['protein_id'])
        if torch.cuda.is_available():
            # test_pbs_vertex = torch.FloatTensor(test_p['protein_feature']).cuda()
            # test_pbs_vertex = torch.FloatTensor(test_p['protein_feature_protbert_bfd']).cuda()
            test_pbs_vertex = torch.FloatTensor(test_p['protein_feature_prottranst5xluf50']).cuda()
        else:
            # test_pbs_vertex = torch.FloatTensor(test_p['protein_feature'])
            # test_pbs_vertex = torch.FloatTensor(test_p['protein_feature_protbert_bfd'])
            test_pbs_vertex = torch.FloatTensor(test_p['protein_feature_prottranst5xluf50'])

        test_pbs_label = torch.LongTensor(test_p['protein_label'])

        p_preds = model(test_pbs_vertex)
        p_preds = p_preds.data.cpu().numpy()
        test_pbs_label = test_pbs_label.numpy()

        all_labels.append(test_pbs_label)
        all_preds.append(p_preds)

        g_auc_roc = compute_auc_roc(test_pbs_label, p_preds)
        g_auc_pr = compute_auc_pr(test_pbs_label, p_preds)
        all_auc_pr.append(g_auc_pr)
        all_auc_roc.append(g_auc_roc)
        
    y_test = np.concatenate(all_labels, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)

    numActualPoses = np.count_nonzero(y_test)
    sortedPreds = np.sort(y_pred)
    cutoffInd = sortedPreds.size - numActualPoses
    thresholdOpt = sortedPreds[cutoffInd]

    w_auc_roc = compute_auc_roc(y_test, y_pred)
    w_auc_pr = compute_auc_pr(y_test, y_pred)

    y_pred[y_pred >= thresholdOpt] = 1
    y_pred[y_pred < thresholdOpt] = 0
    w_acc = compute_acc(y_test, y_pred)
    w_mcc, w_recall, w_precision, w_f1_score, w_sc, w_sp = compute_performance(y_test, y_pred)

    all_w_results = [w_auc_roc, w_auc_pr, w_acc, w_mcc, w_f1_score, w_precision, w_recall, w_sc, w_sp, thresholdOpt]

    return all_results, all_w_results
