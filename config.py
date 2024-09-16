
from arguments import args

cfg = {}

cfg['is_training'] = args.is_training
cfg['model_id'] = args.model_id
cfg['model'] = args.model

# data loader
cfg['data'] = args.data
cfg['root_path'] = args.root_path
cfg['data_path'] = args.data_path
cfg['features'] = args.features
cfg['target'] = args.target
cfg['freq'] = args.freq
cfg['checkpoints'] = args.checkpoints

# forecasting task
cfg['seq_len'] = args.seq_len
cfg['label_len'] = args.label_len
cfg['pred_len'] = args.pred_len

# model define
cfg['bucket_size'] = args.bucket_size
cfg['n_hashes'] = args.n_hashes
cfg['enc_in'] = args.enc_in
cfg['dec_in'] = args.dec_in
cfg['c_out'] = args.c_out
cfg['d_model'] = args.d_model
cfg['n_heads'] = args.n_heads
cfg['e_layers'] = args.e_layers
cfg['d_layers'] = args.d_layers
cfg['d_ff'] = args.d_ff
cfg['moving_avg'] = args.moving_avg
cfg['factor'] = args.factor
cfg['distil'] = args.distil
cfg['dropout'] = args.dropout
cfg['embed'] = args.embed
cfg['activation'] = args.activation
cfg['output_attention'] = args.output_attention
cfg['do_predict'] = args.do_predict

# optimization
cfg['num_workers'] = args.num_workers
cfg['itr'] = args.itr
cfg['train_epochs'] = args.train_epochs
cfg['batch_size'] = args.batch_size
cfg['patience'] = args.patience
cfg['learning_rate'] = args.learning_rate
cfg['des'] = args.des
cfg['loss'] = args.loss
cfg['lradj'] = args.lradj
cfg['use_amp'] = args.use_amp

# GPU
cfg['use_gpu'] = args.use_gpu
cfg['gpu'] = args.gpu
cfg['use_multi_gpu'] = args.use_multi_gpu
cfg['devices'] = args.devices

exec_type = 'train'

if exec_type == 'train':
    cfg['is_training'] = 1
    cfg['root_path'] = './data/'
    cfg['data_path'] = 'weatherAUS_preprocessed'
    cfg['model_id'] = 'AUS_96_96'
    cfg['model'] = 'Autoformer'
    cfg['data'] = 'custom'
    cfg['features'] = 'MS'
    cfg['target'] = 'Rainfall'
    cfg['seq_len'] = 96
    cfg['label_len'] = 48
    cfg['pred_len'] = 96
    cfg['e_layers'] = 2
    cfg['d_layers'] = 1
    cfg['factor'] = 3
    cfg['enc_in'] = 25
    cfg['dec_in'] = 25
    cfg['c_out'] = 1
    cfg['des'] = 'Exp'
    cfg['train_epochs'] = 20
    cfg['itr'] = 1

    args.is_training = 1
    args.root_path = './data/kaggle/'
    args.data_path = 'weatherAUS_preprocessedwo_zeros.csv'
    args.model_id = 'AUS_96_96'
    args.model = 'Autoformer'
    args.data = 'custom'
    args.features = 'MS'
    args.target = 'Rainfall'
    args.seq_len = 96
    args.label_len = 48
    args.pred_len = 96
    args.e_layers = 2
    args.d_layers = 1
    args.factor = 3
    args.enc_in = 25
    args.dec_in = 25
    args.c_out = 1
    args.des = 'Exp'
    args.train_epochs = 20
    args.itr = 1
    
