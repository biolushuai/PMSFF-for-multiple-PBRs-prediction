class DefaultConfig(object):
    fold_train_dataset_path = r'D:\202211\ProjectsData\SeRenDIP-epitope'
    fold_test_dataset_path = r'D:\202211\ProjectsData\SeRenDIP-epitope'

    noval_train_dataset_path = r'D:\202211\ProjectsData\HeteHomoPPI\hhc-train.pkl'
    noval_test_dataset_path = r'D:\202211\ProjectsData\HeteHomoPPI\homo-test.pkl'

    train_dataset_path = r'./data/pecan-paratope-train.pkl'
    val_dataset_path = r'./data/pecan-paratope-val.pkl'
    test_dataset_path = r'./data/pecan-paratope-test.pkl'

    save_path = r'./models_saved/test_ep_hhc'

    epochs = 100
    folds = 5

    learning_rate = 0.001
    weight_decay = 5e-4
    dropout_rate = 0.2
    neg_wt = 0.1
    split_rate = 0.8

    #  input layer
    feature_dim = 1024
    window_padding_sizes = [2, 7, 12]

    # attention layer
    attention_hidden_dim = 128

    # cnn layer
    kernel_sizes = [3, 5, 7]
    out_channels = 1024

    # rnn layer
    rnn_hidden_layer = 1
    rnn_hidden_dim = 64

    # mlp
    mlp_dim = 512

    # training
    early_stop = 10
    seeds = [649737, 395408, 252356, 343053, 743746]
    # seeds = [649737, 395408, 252356, 343053, 743746, 175343, 856516, 474313, 838382, 202003]
