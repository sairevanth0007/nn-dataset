import json
import uuid

from ab.nn.util.Util import *
from ab.nn.util.db.Init import init_db, sql_conn, close_conn


def init_population():
    if not exists(db_file):
        init_db()
        json_n_code_to_db()

def uuid4():
    return str(uuid.uuid4())


def code_to_db(code_file, table_name, cursor):
    nm = code_file.stem
    with open(code_file, 'r') as file:
        model_code = file.read()
    # Check if the model exists in the database
    cursor.execute(f"SELECT code FROM {table_name} WHERE name = ?", (nm,))
    existing_entry = cursor.fetchone()
    if existing_entry:
        # If model exists, update the code if it has changed
        existing_code = existing_entry
        if existing_code != model_code:
            print(f"Updating code for model: {nm}")
            cursor.execute("UPDATE nn SET code = ? WHERE name = ?", (model_code, nm))
    else:
        # If model does not exist, insert it with a new UUID
        nm = nm if nm is not None else uuid4()
        cursor.execute(f"INSERT INTO {table_name} (name, code) VALUES (?, ?)", (nm, model_code))


def populate_code_table(table_name, cursor, name=None):
    """
    Populate the code table with models from the appropriate directory.
    """
    code_dir = nn_path(table_name)
    code_files = [code_dir / f"{name}.py"] if name is not None else [Path(f) for f in code_dir.iterdir() if f.is_file() and f.suffix == '.py' and f.name != '__init__.py']
    for code_file in code_files:
        code_to_db(code_file, table_name, cursor)
    print(f"{table_name} added/updated in the `{table_name}` table: {[f.stem for f in code_files]}")


def populate_prm_table(table_name, cursor, prm, uid):
    """
    Populate the parameter table with variable number of parameters of different types.
    """
    for nm, value in prm.items():
        cursor.execute(f"INSERT INTO {table_name} (uid, name, value, type) VALUES (?, ?, ?, ?)",
                       (uid, nm, str(value), type(value).__name__))


def save_stat(task, dataset, nn, metric, epoch, prm, cursor):
    # Insert each trial into the database with epoch
    transform = prm.pop('transform')
    uid = prm.pop('uid')
    extra_main_column_values = [prm.pop(nm, None) for nm in extra_main_columns]
    for nm in param_tables:
        populate_prm_table(nm, cursor, prm, uid)
    cursor.execute(f"""
    INSERT INTO stat (id, task, dataset, nn, transform, metric, epoch, prm, {', '.join(extra_main_columns)}) 
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (uid, task, dataset, nn, transform, metric, epoch, uid, *extra_main_column_values))


def json_n_code_to_db():
    """
    Reload all statistics into the database for all subconfigs and epochs.
    """
    conn, cursor = sql_conn()
    stat_base_path = Path(stat_dir)
    sub_configs = [d.name for d in stat_base_path.iterdir() if d.is_dir()]

    for sub_config in sub_configs:
        model_stat_dir = stat_base_path / sub_config

        for epoch_file in Path(model_stat_dir).iterdir():
            model_stat_file = model_stat_dir / epoch_file
            epoch = int(epoch_file.stem)

            with open(model_stat_file, 'r') as f:
                trials = json.load(f)

            for trial in trials:
                task, dataset, metric, nn = conf_to_names(sub_config)
                populate_code_table('nn', cursor, name=nn)
                populate_code_table('metric', cursor, name=metric)
                populate_code_table('transform', cursor, name=trial['transform'])
                save_stat(task, dataset, nn, metric, epoch, trial, cursor)
    close_conn(conn)
    print("All statistics reloaded successfully.")


def save_results(config: str, epoch: int, model_stat_file: str, prm: dict):
    """
    Save Optuna study results for a given model in JSON-format.
    :param config: Config (Task, Dataset, Metric, and Model name).
    :param epoch: Number of epochs.
    :param model_stat_file: File for the model statistics.
    :param prm: Dictionary of all saved parameters.
    """
    task, dataset, metric, nn = conf_to_names(config)
    # Save results to SQLite DB
    conn, cursor = sql_conn()
    save_stat(task, dataset, nn, metric, epoch, prm, cursor)
    close_conn(conn)


def save_nn(nn_code : str, task : str, dataset : str, metric : str):
    name = uuid4()
    conn, cursor = sql_conn()
    # todo: implement saving of NN model into database (its name generated with uuid4())
    close_conn(conn)
    return name

init_population()
