from ab.nn.util.db.Write import init_population

init_population()

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
