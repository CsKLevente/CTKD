import os
import json
from collections import namedtuple
import pandas as pd
import psycopg2


INSERT_SQL = '''
INSERT INTO public.experiment_data(
    dirpath, raw_config, raw_metrics, models, final_stats, pruned_metrics)
    VALUES (%(dirpath)s, %(config)s, %(metrics)s::json[],  %(models)s, %(final_stats)s, %(pruned_metrics)s)
ON CONFLICT DO NOTHING
RETURNING experiment_data_id;
'''

DATABASE = "postgres://admin:HvN3TX4VdTP4gOftIGyQsCxuBdcyRhvJ@dpg-cnl3dquv3ddc73f659sg-a.frankfurt-postgres.render.com/thesis_kxlg"

ROOT = "./save/student_model/shrinkage_strength"
EXPERIMENT_DATA_FILES = ['test_best_metrics.json', 'config.json', 'final_pruned_metrics.json']
SAVE_CSV_PATH = "./save/ExperimentData.csv"
SAVE_TO_FILE = False

EXP = "./save/student_model/imagenette-half_size-tea-res34-stu-res8/KD/sparsity_learning/kd_T_4"
EXP_SAVE_CSV_PATH = "./save/ExperimentMetricsHistory.csv"

Experiment_data = namedtuple('Experiment', ('dirpath', 'config', 'metrics', 'models', 'final_stats','pruned_metrics'))
History_data = namedtuple('History',('epoch','test_accuracy'))


def db_connect(db_string=DATABASE):
    """Get Database connection

    @param db_string: Database connection string
    @return: Database connection
    """
    return psycopg2.connect(db_string)


def insert_data_to_db(connection, sql, data_list):
    with connection:
        with connection.cursor() as curs:
            for data in data_list:
                curs.execute(sql, data)
            records = [record for record in curs]
    print(records)


def prepare_experiments_data_for_sql_insert(experiments):
    from psycopg2.extras import Json
    for experiment in experiments:
        exp = experiment._asdict()
        exp["config"] = Json(exp["config"])
        exp["metrics"] = [Json(metric) for metric in exp["metrics"]]
        exp["final_stats"] = Json(exp["final_stats"])
        exp["pruned_metrics"] = Json(exp["pruned_metrics"])
        yield exp

    # sql_data = []
    # for experiment in experiments:
    #     data = experiment._asdict()


def iterate_experiment_folders(root=ROOT):
    for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        if dirnames:
            continue
        if len(filenames) < 3:
            continue
        if not any(fname.endswith('.pth') for fname in filenames):
            continue
        if not all(files in filenames for files in EXPERIMENT_DATA_FILES):
            continue
        yield dirpath, filenames


def collect_data_from_experiment_folder(dirpath, filenames):
    # print(dirpath, filenames)
    metrics_path = os.path.join(dirpath, EXPERIMENT_DATA_FILES[0])
    # print(metrics_path)
    with open(metrics_path) as f:
        metrics = json.loads('[' + f.read().replace('}{', '},{') + ']')
    # print(len(metrics), metrics[-1])
    config_path = os.path.join(dirpath, EXPERIMENT_DATA_FILES[1])
    with open(config_path) as f:
        config = json.load(f)
    # print(config)
    pruned_metrics_path = os.path.join(dirpath, EXPERIMENT_DATA_FILES[2])
    with open(pruned_metrics_path) as f:
        pruned_metrics = json.load(f)
    model_saves = []
    for filename in filenames:
        if filename in EXPERIMENT_DATA_FILES:
            continue
        model_saves.append(filename)
    # print(model_saves)
    return Experiment_data(dirpath=dirpath, config=config, metrics=metrics, models=model_saves, final_stats=metrics[-1], pruned_metrics=pruned_metrics )


def collect_metrics(dirpath, filenames):
    metrics_path = os.path.join(dirpath, EXPERIMENT_DATA_FILES[0])
    # print(metrics_path)
    with open(metrics_path) as f:
        metrics = json.loads('[' + f.read().replace('}', '},').rstrip(',') + ']')
    history = [History_data(metric['epoch'],metric['test_acc']) for metric in metrics]
    return history

def main():
    experiments = [collect_data_from_experiment_folder(dirpath, filenames)
                   for dirpath, filenames in iterate_experiment_folders(ROOT)]

    print(experiments[0]._asdict())

    connection = db_connect(DATABASE)
    try:
        print(connection)
        insert_data_to_db(connection, INSERT_SQL, prepare_experiments_data_for_sql_insert(experiments))
    finally:
        connection.close()

    df = pd.DataFrame(
        experiments
    )
        # data_nt = collect_data_from_experiment_folder(dirpath, filenames)
        # print(data_nt.dirpath, data_nt.config)
    # print(df.config[0])

    df_flat = df.drop('config', axis=1).join(pd.json_normalize(df.config).add_prefix('config_'))
    df_flat['student'] = df_flat.dirpath.str.contains('student')
    df_flat['test_accuracy'] = df_flat.final_stats.apply(lambda x: x['test_acc'])
    df_flat['best_accuracy'] = df_flat.final_stats.apply(lambda x: x.get('best_acc'))
    print(df_flat.columns)
    if SAVE_TO_FILE:
        df_flat[['dirpath', 'models','test_accuracy', 'best_accuracy', 'student',
           'config_batch_size', 'config_num_workers', 'config_epochs',
           'config_experiments_dir', 'config_experiments_name',
           'config_learning_rate', 'config_lr_decay_epochs',
           'config_lr_decay_rate', 'config_weight_decay', 'config_momentum',
           'config_dataset', 'config_model_s', 'config_path_t', 'config_path_s',
           'config_distill', 'config_gamma', 'config_alpha',
           'config_beta', 'config_kd_T', 'config_have_mlp', 'config_mlp_name',
           'config_t_start', 'config_t_end', 'config_cosine_decay',
           'config_decay_max', 'config_decay_min', 'config_decay_loops',
           'config_dkd_alpha', 'config_dkd_beta', 'config_mode',
           'config_nce_k', 'config_nce_t', 'config_nce_m', 'config_save_model',
           'config_no_edge_transform', 'config_multiprocessing_distributed',
           'config_deterministic',
           'config_sparsity_learning',
           'config_model_path', 'config_model_t', 'config_half_size_img', 'config_save_freq',
           'config_model' ]].to_csv(SAVE_CSV_PATH)
if __name__ == '__main__':
    main()
    # df = None
    # for i, (dirpath, filenames) in enumerate(iterate_experiment_folders(EXP)):
    #     df_fold = pd.DataFrame(collect_metrics(dirpath, filenames))
    #     print(df_fold)
    #     if df is None:
    #         df = df_fold.copy(deep=True)
    #     else:
    #         df = df.join(df_fold, rsuffix=str(i))
    # print(df)
    # print(df.columns)
    # df[['epoch', 'test_accuracy', 'test_accuracy1',
    #    'test_accuracy2', 'test_accuracy3',
    #    'test_accuracy4']].to_csv(EXP_SAVE_CSV_PATH)







