import json
from os import listdir, makedirs
import os
from ab.nn.util.Util import *
from ab.nn.util.db.Write import init_population

init_population()


def data(only_best_accuracy=False, task=None, dataset=None, metric=None, nn=None, epoch=None) -> tuple[dict[str, int | float | str | dict[str, int | float | str]], ...]:
    # todo: update this function to load data from dataset
    """
    Get the NN model code and all related statistics
    :param only_best_accuracy: if True returns for every NN model only statistics for the best accuracy, else all collected statistics
    :param task: Name of the task. If provided, it is used to filter the data.
    :param dataset: Name of dataset. If provided, it is used to filter the data.If provided, it is used to filter the data.
    :param metric: Name of the NN performance metric. If provided, it is used to filter the data.
    :param nn: Neural network name. If provided, it is used to filter the data.
    :param epoch: Epoch of the NN training process. If provided, it is used to filter the data.

    Returns: Tuple of dictionaries. Each dictionary has the following keys with types
                                  {'task': str, 'dataset': str,
                                   'metric': str, 'metric_code': str,
                                   'nn': str, 'nn_code': str,
                                   'epoch': int, 'accuracy': float,
                                   'duration': int,
                                   'prm': dict,
                                   'transform_code': <transformer code: str>}
    where
    - duration is duration of the training in milliseconds
    - prm is dict[str, int | float | str] of NN training hyperparameters including, but not limiting to
    following keys:
    {
        'batch': 4,
        'dropout': 0.17920158482473114,
        'lr': 0.02487720458587122,
        'momentum': 0.3867297180491852,
        'transform': 'norm_299'
    }
    """

    out = []
    for stat_folder in listdir(stat_dir):
        curr_task, curr_dataset, curr_metric, nn = conf_to_names(stat_folder)

        with open(metric_dir / (curr_metric + '.py'), 'r') as code_file:
            metric_code = str(code_file.read())

        with open(nn_dir / (nn + '.py'), 'r') as code_file:
            nn_code = str(code_file.read())

        for epoch_file in listdir(stat_dir / stat_folder):
            with open(stat_dir / stat_folder / epoch_file, 'r') as json_file:
                content = json.load(json_file)

            for stat in content:
                with open(transform_dir / (stat['transform'] + '.py'), 'r') as code_file:
                    transform_code = str(code_file.read())
                stat.pop('uid')
                next_row = {
                    'task': curr_task,
                    'dataset': curr_dataset,
                    'metric': curr_metric,
                    'metric_code': metric_code,
                    'nn': nn,
                    'nn_code': nn_code,
                    'epoch': Path(epoch_file).stem,
                    'accuracy': stat.pop('accuracy'),
                    'duration': stat.pop('duration', None),
                    'prm': stat,
                    'transform_code': transform_code
                }
                out.append(next_row)
                if only_best_accuracy: break
    return tuple(out)


def remaining_trials(trial_file, model_name, n_optuna_trials):
    """
    Calculates the remaining Optuna trials based on the completed ones. Checks for a 'trials.json' file in the
    specified directory to determine how many trials have been completed, and returns the number of trials left.
    :param trial_file: Trial file path
    :param model_name: Name of the model.
    :param n_optuna_trials: The total number of Optuna trials the model should have. If negative, its absolute value represents the number of additional trials.
    :return: Number of remaining trials.
    """
    n_passed_trials = 0
    if trial_file.exists():
        with open(trial_file, 'r') as f:
            trials = json.load(f)
            n_passed_trials = len(trials)
    n_remaining_trials = abs(n_optuna_trials) if n_optuna_trials < 0 else max(0, n_optuna_trials - n_passed_trials)
    if n_passed_trials > 0:
        print(f"The {model_name} passed {n_passed_trials} trials, {n_remaining_trials} left.")
    return n_remaining_trials, n_passed_trials


def supported_transformers():
    return ['complex','complex_32','complex_64','complex_128','complex_256','complex_299', 'complex_512',
            'norm', 'norm_32', 'norm_64', 'norm_128', 'norm_256', 'norm_299', 'norm_512',
            'echo', 'echo_32', 'echo_64', 'echo_128', 'echo_256', 'echo_299', 'echo_512'
            'complex_flip', 'complex_32_flip', 'complex_64_flip', 'complex_128_flip', 'complex_256_flip', 'complex_299_flip', 'complex_512_flip',
            'norm_flip', 'norm_32_flip', 'norm_64_flip', 'norm_128_flip', 'norm_256_flip', 'norm_299_flip', 'norm_512_flip',
            'echo_flip', 'echo_32_flip', 'echo_64_flip', 'echo_128_flip', 'echo_256_flip', 'echo_299_flip', 'echo_512_flip']


# todo: Request from the database unique names of all configures corresponding to config-patterns
# once the database is loaded, the function will be updated

def unique_configs(patterns: tuple[str, ...]) -> list[str]:
    """
    Collect models matching the given configuration prefix.
    :param patterns: Tuple of configuration prefixes.
    :return: List of unique configuration directories.
    """
    all_configs = []
    for pattern in patterns:
        # Collect configurations matching the pattern
        matching_configs = [config for config in os.listdir(stat_dir) if config.startswith(pattern)]
        # If no match, create a directory if the pattern is a full configuration
        if not matching_configs and is_full_config(pattern):
            makedirs(stat_dir / pattern)
            matching_configs = unique_configs((pattern,))
        all_configs.extend(matching_configs)
    return list(set(all_configs))

from ab.nn.util.Util import *
from ab.nn.util.db.Write import init_population
from ab.nn.util.db.Init import sql_conn, close_conn
from ab.nn.util.Const import config_splitter

init_population()


import json
from ab.nn.util.db.Write import init_population
from ab.nn.util.Const import config_splitter  # You already use this
from ab.nn.util.db.Init import sql_conn, close_conn

init_population()

def data(only_best_accuracy=False, task=None, dataset=None, metric=None, nn=None, epoch=None) -> tuple[dict[str, int | float | str | dict[str, int | float | str]], ...]:
    """
    Get the NN model code and all related statistics from the database.
    
    - If only_best_accuracy == True, then for every unique combination of 
      (task, dataset, metric, nn, epoch) only the row with the highest accuracy is returned.
    - If only_best_accuracy == False, all matching rows are returned.
    - Additionally, if any of the parameters (task, dataset, metric, nn, epoch) is not None,
      the results are filtered accordingly in the SQL query.
      
    Each returned dictionary has the following keys:
      - 'task': str
      - 'dataset': str
      - 'metric': str
      - 'metric_code': str    (source code from the metric table)
      - 'nn': str
      - 'nn_code': str        (source code from the nn table)
      - 'epoch': int
      - 'accuracy': float
      - 'duration': int
      - 'prm': dict           (hyperparameters, reconstructed from the "prm" table)
      - 'transform_code': str (source code from the transform table)
    """
    
    # Build filtering conditions based on provided parameters.
    filters = []
    params = []
    if task is not None:
        filters.append("s.task = ?")
        params.append(task)
    if dataset is not None:
        filters.append("s.dataset = ?")
        params.append(dataset)
    if metric is not None:
        filters.append("s.metric = ?")
        params.append(metric)
    if nn is not None:
        filters.append("s.nn = ?")
        params.append(nn)
    if epoch is not None:
        filters.append("s.epoch = ?")
        params.append(epoch)
    where_clause = "WHERE " + " AND ".join(filters) if filters else ""
    
    # Get a connection and cursor from the database infrastructure.
    conn, cursor = sql_conn()
    
    if not only_best_accuracy:
        # Query that returns all matching rows.
        query = f"""
            SELECT s.task, s.dataset, s.metric, m.code AS metric_code,
                   s.nn, n.code AS nn_code, s.epoch, s.accuracy, s.duration,
                   s.id AS stat_id, s.prm AS stat_prm, t.code AS transform_code, s.transform
            FROM stat s
            LEFT JOIN nn n ON s.nn = n.name
            LEFT JOIN metric m ON s.metric = m.name
            LEFT JOIN transform t ON s.transform = t.name
            {where_clause}
            ORDER BY s.task, s.dataset, s.metric, s.nn, s.epoch;
        """
        cursor.execute(query, params)
        rows = cursor.fetchall()
    else:
        # Query that returns, for each group (task, dataset, metric, nn, epoch),
        # only the row with the maximum accuracy.
        query = f"""
            WITH filtered AS (
                SELECT s.task, s.dataset, s.metric, s.nn, s.transform, s.epoch, s.accuracy, s.duration,
                       s.id AS stat_id, s.prm AS stat_prm
                FROM stat s
                {where_clause}
            ),
            best AS (
                SELECT task, dataset, metric, nn, epoch, MAX(accuracy) AS max_accuracy
                FROM filtered
                GROUP BY task, dataset, metric, nn, epoch
            )
            SELECT f.task, f.dataset, f.metric, m.code AS metric_code,
                   f.nn, n.code AS nn_code, f.epoch, f.accuracy, f.duration,
                   f.stat_id, f.stat_prm, t.code AS transform_code, f.transform
            FROM filtered f
            JOIN best b ON f.task = b.task 
                        AND f.dataset = b.dataset 
                        AND f.metric = b.metric 
                        AND f.nn = b.nn 
                        AND f.epoch = b.epoch 
                        AND f.accuracy = b.max_accuracy
            LEFT JOIN nn n ON f.nn = n.name
            LEFT JOIN metric m ON f.metric = m.name
            LEFT JOIN transform t ON f.transform = t.name
            ORDER BY f.task, f.dataset, f.metric, f.nn, f.epoch;
        """
        cursor.execute(query, params)
        rows = cursor.fetchall()
    
    # Extract column names from cursor description.
    columns = [col[0] for col in cursor.description]
    
    results = []
    # For each row from the main stat table, reconstruct the hyperparameter dictionary
    # by querying the "prm" table using the stat's unique id.
    for row in rows:
        row_dict = dict(zip(columns, row))
        uid = row_dict.get("stat_id")
        prm_dict = {}
        # Query the prm table to get hyperparameters for this stat record.
        cursor.execute("SELECT name, value, type FROM prm WHERE uid = ?", (uid,))
        prm_rows = cursor.fetchall()
        for pr in prm_rows:
            # Each pr is a tuple (name, value, type)
            pr_name, pr_value, pr_type = pr
            # Convert the stored value to its proper type.
            if pr_type == "int":
                try:
                    pr_value = int(pr_value)
                except Exception:
                    pass
            elif pr_type == "float":
                try:
                    pr_value = float(pr_value)
                except Exception:
                    pass
            prm_dict[pr_name] = pr_value
        row_dict["prm"] = prm_dict
        
        # Remove internal columns not needed in the output.
        row_dict.pop("stat_id", None)
        row_dict.pop("stat_prm", None)
        row_dict.pop("transform", None)  # Only transform_code is needed.
        
        # Ensure that epoch is an integer.
        try:
            row_dict["epoch"] = int(row_dict["epoch"])
        except (ValueError, TypeError):
            pass
        
        results.append(row_dict)
    
    close_conn(conn)
    return tuple(results)

def remaining_trials(trial_file, model_name, n_optuna_trials) -> tuple[int, int]:
    """
    Calculate the number of remaining Optuna trials for a given model configuration by querying the database.
    
    Instead of reading trial counts from a file, we query the database to count all trial records
    for the specified model (identified by model_name). The trial_file parameter is retained for
    interface compatibility but is not used.
    
    If n_optuna_trials is negative, its absolute value is taken as the required number of additional trials.
    Otherwise, the function computes:
    
        remaining_trials = max(0, n_optuna_trials - n_passed_trials)
    
    :param trial_file: A file path that used to store trial data (not used in this database-based implementation).
    :param model_name: The name of the model configuration (used to filter the trials).
    :param n_optuna_trials: The target number of trials. If negative, its absolute value specifies the additional trials required.
    :return: A tuple (n_remaining_trials, n_passed_trials) where:
             - n_remaining_trials is the number of new trials to run (or 0 if none remain).
             - n_passed_trials is the number of trials already recorded in the database for this model.
    """

    conn, cursor = sql_conn()
    
    query = "SELECT COUNT(*) AS trial_count FROM stat WHERE nn = ?"
    cursor.execute(query, (model_name,))
    row = cursor.fetchone()
    if row:
        # Convert the tuple row to a dict
        columns = [col[0] for col in cursor.description]
        row_dict = dict(zip(columns, row))
        n_passed_trials = row_dict.get("trial_count", 0)
    else:
        n_passed_trials = 0

    if n_optuna_trials < 0:
        n_remaining_trials = abs(n_optuna_trials)
    else:
        n_remaining_trials = max(0, n_optuna_trials - n_passed_trials)

    if n_passed_trials > 0:
        print(f"Model '{model_name}' has {n_passed_trials} recorded trial(s), {n_remaining_trials} remaining.")

    close_conn(conn)
    return n_remaining_trials, n_passed_trials



def supported_transformers() -> list[str]:
    """
    Returns a list of all transformer names available in the database.
    
    The function queries the 'transform' table for all records and extracts the 'name'
    field from each row.
    """

    conn, cursor = sql_conn()
    query = "SELECT name FROM transform"
    cursor.execute(query)
    rows = cursor.fetchall()
    # Since each row is a tuple (with one element), you can simply use row[0]
    transformer_names = [row[0] for row in rows]
    close_conn(conn)
    return transformer_names



# todo: Request from the database unique names of all configures corresponding to config-patterns
# once the database is loaded, the function will be updated

def unique_configs(patterns: tuple[str, ...]) -> list[str]:
    """
    Returns a list of unique configuration strings from the database that match at least one of the input patterns.
    
    A configuration string is constructed by concatenating the 'task', 'dataset', 'metric', and 'nn'
    fields from the 'stat' table using the configuration splitter defined in your constants.
    
    :param patterns: A tuple of configuration prefix patterns.
    :return: A list of unique configuration strings that start with any of the provided patterns.
    """

    conn, cursor = sql_conn()
    query = "SELECT DISTINCT task, dataset, metric, nn FROM stat"
    cursor.execute(query)
    rows = cursor.fetchall()
    columns = [col[0] for col in cursor.description]
    close_conn(conn)
    
    all_configs = []
    for row in rows:
        row_dict = dict(zip(columns, row))
        config_str = f"{row_dict['task']}{config_splitter}{row_dict['dataset']}{config_splitter}{row_dict['metric']}{config_splitter}{row_dict['nn']}"
        all_configs.append(config_str)
    
    matched_configs = []
    for pattern in patterns:
        for config in all_configs:
            if config.startswith(pattern):
                matched_configs.append(config)
    
    return list(set(matched_configs))
