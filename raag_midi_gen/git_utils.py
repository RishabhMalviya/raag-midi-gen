import subprocess
from datetime import datetime

from mlflow.entities import Run

from raag_midi_gen.paths import EXPERIMENT_LOGS_DIR


class GitOutOfSyncError(Exception):
    pass


def _git(*args):
    return subprocess.run(["git"] + list(args), check=True, capture_output=True, text=True).stdout.strip()


def _get_current_hash():
    return _git("rev-parse", "HEAD")


def check_repo_is_in_sync():
    are_un_tracked_changes = not (len(_git("ls-files", "--others", "--exclude-standard")) == 0)
    if are_un_tracked_changes:
        raise GitOutOfSyncError('You have un-tracked changes.\n'
                                'Make sure you are in sync with the remote Git repo before running an experiment.')

    are_un_committed_changes = not (len(_git("diff")) == 0)
    if are_un_committed_changes:
        raise GitOutOfSyncError('You have un-committed changes.\n'
                                'Make sure you are in sync with the remote Git repo before running an experiment.')

    currently_tracked_remote_branch = _git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
    are_un_pushed_changes = not (len(_git("diff", f'{currently_tracked_remote_branch}..HEAD')) == 0)
    if are_un_pushed_changes:
        raise GitOutOfSyncError('You have un-pushed changes.\n'
                                'Make sure you push all changes to the remote repo before running an experiment')

    return _get_current_hash()


def commit_latest_run(experiment_name, mlflow_run: Run = None):
    run_name = mlflow_run.info.run_name
    end_time =  datetime.fromtimestamp(mlflow_run.info.end_time / 1000.0).isoformat()  # MLFlow end_time is milliseconds since UNIX epoch

    commit_message = f'Log run {run_name} under {experiment_name}, completed on {end_time}'

    git_add_output = _git("add", EXPERIMENT_LOGS_DIR, '.')
    print(git_add_output)

    git_commit_output = _git("commit", '-m', commit_message)
    print(git_commit_output)

    git_push_output = _git("push")
    print(git_push_output)