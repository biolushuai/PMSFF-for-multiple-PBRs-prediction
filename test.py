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

    # 最佳阈值下的模型性能
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

    for (p_pred, p_label, protein) in zip(all_preds, all_labels, test_proteins):
        p_pred[p_pred >= thresholdOpt] = 1
        p_pred[p_pred < thresholdOpt] = 0

        g_acc = compute_acc(p_label, p_pred)
        g_mcc, g_recall, g_precision, g_f1_score, g_sc, g_sp = compute_performance(p_label, p_pred)

        all_acc.append(g_acc)
        all_recall.append(g_recall)
        all_precision.append(g_precision)
        all_mcc.append(g_mcc)
        all_f_score.append(g_f1_score)
        all_sc.append(g_sc)
        all_sp.append(g_sp)

    all_results = [np.mean(all_auc_roc), np.mean(all_auc_pr), np.mean(all_acc), np.mean(all_mcc), np.mean(all_f_score),
                   np.mean(all_precision), np.mean(all_recall), np.mean(all_sc), np.mean(all_sp), thresholdOpt]

    all_w_results = [w_auc_roc, w_auc_pr, w_acc, w_mcc, w_f1_score, w_precision, w_recall, w_sc, w_sp, thresholdOpt]

    # return all_results, all_w_results
    return all_auc_roc, all_auc_pr, all_results, all_w_results


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    models_saved_path = r'./models_saved/models_saved_best_t5xluf'
    models = os.listdir(models_saved_path)
    test_path = r'D:\ProjectsData\MsPBRsP\pipenn-zk448-a-transt5xluf50.pkl'

    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)

    from models import A3C3GRUModel
    test_model = A3C3GRUModel()

    # from utils import process_data
    # process_data(test_data)

    # with open(os.path.join(models_saved_path, 'average_results_equal.csv'), 'w', encoding='utf-8', newline='') as f:
    #     csv_writer = csv.writer(f)
    #     csv_writer.writerow(['auc_roc', 'auc_pr', 'acc', 'mcc_max', 'f1', 'p', 'r', 'sc', 'sp', 't_max'])
    #
    # with open(os.path.join(models_saved_path, 'whole_results_equal.csv'), 'w', encoding='utf-8', newline='') as f:
    #     csv_writer = csv.writer(f)
    #     csv_writer.writerow(['auc_roc', 'auc_pr', 'acc', 'mcc_max', 'f1', 'p', 'r', 'sc', 'sp', 't_max'])

    num = 1
    for m in models:
        if '.tar' in m:
            test_model_path = os.path.join(models_saved_path, m)
            print(test_model_path)
            test_model_sd = torch.load(test_model_path)
            test_model.load_state_dict(test_model_sd)
            if torch.cuda.is_available():
                model = test_model.cuda()

            # experiment_results, whole_results = test(test_model, test_data)
            a_auc, a_pr, experiment_results, whole_results = test(test_model, test_data)
            with open(os.path.join(models_saved_path, 'm{}_zk448_auc.txt'.format(num)), 'w') as f:
                for p_id, i in enumerate(a_auc):
                    f.write(test_data[p_id]['protein_id'] + ' ' + str(i) + '\n')

            with open(os.path.join(models_saved_path, 'm{}_zk448_pr.txt'.format(num)), 'w') as f:
                for p_id, j in enumerate(a_pr):
                    f.write(test_data[p_id]['protein_id'] + ' ' + str(j) + '\n')
            num += 1

            # with open(os.path.join(models_saved_path, 'average_results_equal.csv'), 'a+', encoding='utf-8', newline='') as f:
            #     csv_writer = csv.writer(f)
            #     csv_writer.writerow(experiment_results)
            #
            # with open(os.path.join(models_saved_path, 'whole_results_equal.csv'), 'a+', encoding='utf-8', newline='') as f:
            #     csv_writer = csv.writer(f)
            #     csv_writer.writerow(whole_results)
