"""Utilities for mangling config vars."""

import copy
import functools
import os
import pathlib
from typing import List, Optional, Union, Iterable, Tuple, Dict, TypeVar
from types import SimpleNamespace, ModuleType

import numpy as np
import mne
import mne_bids
from mne_bids import BIDSPath

from ._logging import logger, gen_log_kwargs
from ._typing import Literal, ArbitraryContrast, _keys_arbitrary_contrast


def get_deriv_root(config: SimpleNamespace) -> pathlib.Path:
    if config.deriv_root is None:
        return get_bids_root(config) / 'derivatives' / config.PIPELINE_NAME
    else:
        return (pathlib.Path(config.deriv_root)
                .expanduser()
                .resolve())


def get_fs_subjects_dir(config: SimpleNamespace) -> pathlib.Path:
    if not config.subjects_dir and config.deriv_root is not None:
        # We do this check here (and not in our regular checks section) to
        # avoid an error message when a user doesn't intend to run the source
        # analysis steps anyway.
        raise ValueError(
            'When specifying a "deriv_root", you must also supply a '
            '"subjects_dir".'
        )

    if not config.subjects_dir:
        return (
            get_bids_root(config) / 'derivatives' / 'freesurfer' / 'subjects')
    else:
        return (pathlib.Path(config.subjects_dir)
                .expanduser()
                .resolve())


def get_fs_subject(
    config: SimpleNamespace,
    subject: str
) -> str:
    subjects_dir = get_fs_subjects_dir(config)

    if config.use_template_mri is not None:
        return config.use_template_mri

    if (pathlib.Path(subjects_dir) / subject).exists():
        return subject
    else:
        return f'sub-{subject}'


def get_bids_root(config: SimpleNamespace) -> pathlib.Path:
    # BIDS_ROOT environment variable takes precedence over any configuration
    # file values.
    root = os.getenv('BIDS_ROOT')
    if root is not None:
        return (pathlib.Path(root)
                .expanduser()
                .resolve(strict=True))

    # If we don't have a bids_root until now, raise an exception as we cannot
    # proceed.
    if not config.bids_root:
        msg = ('You need to specify `bids_root` in your configuration, or '
               'define an environment variable `BIDS_ROOT` pointing to the '
               'root folder of your BIDS dataset')
        raise ValueError(msg)

    return (pathlib.Path(config.bids_root)
            .expanduser()
            .resolve(strict=True))


@functools.lru_cache(maxsize=None)
def _get_entity_vals_cached(*args, **kwargs) -> List[str]:
    return mne_bids.get_entity_vals(*args, **kwargs)


def get_datatype(config: SimpleNamespace) -> Literal['meg', 'eeg']:
    # Content of ch_types should be sanitized already, so we don't need any
    # extra sanity checks here.
    if config.data_type is not None:
        return config.data_type
    elif config.data_type is None and config.ch_types == ['eeg']:
        return 'eeg'
    elif config.data_type is None and any(
            [t in ['meg', 'mag', 'grad'] for t in config.ch_types]):
        return 'meg'
    else:
        raise RuntimeError("This probably shouldn't happen. Please contact "
                           "the MNE-BIDS-pipeline developers. Thank you.")


def _get_ignore_datatypes(config: SimpleNamespace) -> Tuple[str]:
    _all_datatypes: List[str] = mne_bids.get_datatypes(
        root=get_bids_root(config)
    )
    _ignore_datatypes = set(_all_datatypes) - set([get_datatype(config)])
    return tuple(sorted(_ignore_datatypes))


def get_subjects(config: SimpleNamespace) -> List[str]:
    env = os.environ
    _valid_subjects = _get_entity_vals_cached(
        root=get_bids_root(config),
        entity_key='subject',
        ignore_datatypes=_get_ignore_datatypes(config),
    )
    if env.get('MNE_BIDS_STUDY_SUBJECT'):
        env_subject = env['MNE_BIDS_STUDY_SUBJECT']
        if env_subject not in _valid_subjects:
            raise ValueError(
                f'Invalid subject. It can be {_valid_subjects} but '
                f'got {env_subject}')
        s = [env_subject]
    elif config.subjects == 'all':
        s = _valid_subjects
    else:
        s = config.subjects

    subjects = set(s) - set(config.exclude_subjects)
    # Drop empty-room subject.
    subjects = subjects - set(['emptyroom'])

    return sorted(subjects)


def get_sessions(
    config: SimpleNamespace
) -> Union[List[None], List[str]]:
    sessions = copy.deepcopy(config.sessions)
    _all_sessions = _get_entity_vals_cached(
        root=get_bids_root(config),
        entity_key='session',
        ignore_datatypes=_get_ignore_datatypes(config),
    )
    env = os.environ
    if env.get('MNE_BIDS_STUDY_SESSION'):
        sessions = env['MNE_BIDS_STUDY_SESSION']
    elif sessions == 'all':
        sessions = _all_sessions

    if not sessions:
        return [None]
    else:
        return sessions


def get_runs_all_subjects(
    config: SimpleNamespace
) -> Dict[str, Union[List[None], List[str]]]:
    """Gives the mapping between subjects and their runs.

    Returns
    -------
    a dict of runs present in the bids_path
    for each subject asked in the configuration file
    (and not for each subject present in the bids_path).
    """
    # We cannot use get_subjects() because if there is just one subject
    subj_runs = dict()
    for subject in get_subjects(config):
        # Only traverse through the current subject's directory
        valid_runs_subj = _get_entity_vals_cached(
            get_bids_root(config) / f'sub-{subject}', entity_key='run',
            ignore_datatypes=_get_ignore_datatypes(config),
        )

        # If we don't have any `run` entities, just set it to None, as we
        # commonly do when creating a BIDSPath.
        if not valid_runs_subj:
            valid_runs_subj = [None]

        if subject in (config.exclude_runs or {}):
            valid_runs_subj = [r for r in valid_runs_subj
                               if r not in config.exclude_runs[subject]]
        subj_runs[subject] = valid_runs_subj

    return subj_runs


def get_intersect_run(config: SimpleNamespace) -> List[str]:
    """Returns the intersection of all the runs of all subjects."""
    subj_runs = get_runs_all_subjects(config)
    return list(set.intersection(*map(set, subj_runs.values())))


def get_runs(
    *,
    config: SimpleNamespace,
    subject: str,
    verbose: bool = False
) -> Union[List[str], List[None]]:
    """Returns a list of runs in the BIDS input data.

    Parameters
    ----------
    subject
        Returns a list of the runs of this subject.
    verbose
        Notify if different subjects do not share the same runs.

    Returns
    -------
    The list of runs of the subject. If no BIDS `run` entity could be found,
    returns `[None]`.
    """
    if subject == 'average':  # Used when creating the report
        return [None]

    runs = copy.deepcopy(config.runs)

    subj_runs = get_runs_all_subjects(config)
    valid_runs = subj_runs[subject]

    if len(get_subjects(config)) > 1:
        # Notify if different subjects do not share the same runs

        same_runs = True
        for runs_sub_i in subj_runs.values():
            if set(runs_sub_i) != set(list(subj_runs.values())[0]):
                same_runs = False

        if not same_runs and verbose:
            msg = ('Extracted all the runs. '
                   'Beware, not all subjects share the same '
                   'set of runs.')
            logger.info(**gen_log_kwargs(message=msg))

    env_run = os.environ.get('MNE_BIDS_STUDY_RUN')
    if env_run and env_run not in valid_runs:
        raise ValueError(
            f'Invalid run. It can be {valid_runs} but '
            f'got {env_run}')
    elif env_run:
        runs = [env_run]
    elif runs == 'all':
        runs = valid_runs

    if not runs:
        return [None]
    else:
        inclusion = set(runs).issubset(set(valid_runs))
        if not inclusion:
            raise ValueError(
                f'Invalid run. It can be a subset of {valid_runs} but '
                f'got {runs}')
        return runs


def get_mf_reference_run(config: SimpleNamespace) -> str:
    # Retrieve to run identifier (number, name) of the reference run
    if config.mf_reference_run is not None:
        return config.mf_reference_run
    # Use the first run
    inter_runs = get_intersect_run(config)
    mf_ref_error = (
        (config.mf_reference_run is not None) and
        (config.mf_reference_run not in inter_runs)
    )
    if mf_ref_error:
        msg = (f'You set mf_reference_run={config.mf_reference_run}, but your '
               f'dataset only contains the following runs: {inter_runs}')
        raise ValueError(msg)
    if inter_runs:
        return inter_runs[0]
    else:
        raise ValueError(
            f"The intersection of runs by subjects is empty. "
            f"Check the list of runs: "
            f"{get_runs_all_subjects()}"
        )


def get_task(config: SimpleNamespace) -> Optional[str]:
    env = os.environ
    task = None
    _valid_tasks = _get_entity_vals_cached(
        root=get_bids_root(config),
        entity_key='task',
        ignore_datatypes=_get_ignore_datatypes(config),
    )
    if env.get('MNE_BIDS_STUDY_TASK'):
        task = env['MNE_BIDS_STUDY_TASK']
        if task not in _valid_tasks:
            raise ValueError(f'Invalid task. It can be: '
                             f'{", ".join(_valid_tasks)} but got: {task}')
    if not task:
        if not _valid_tasks:
            return None
        else:
            return _valid_tasks[0]
    else:
        return task


def get_channels_to_analyze(
    info: mne.Info,
    config: SimpleNamespace
) -> List[str]:
    # Return names of the channels of the channel types we wish to analyze.
    # We also include channels marked as "bad" here.
    # `exclude=[]`: keep "bad" channels, too.
    if get_datatype(config) == 'meg' and _meg_in_ch_types(config.ch_types):
        pick_idx = mne.pick_types(info, eog=True, ecg=True, exclude=[])

        if 'mag' in config.ch_types:
            pick_idx = np.concatenate(
                [pick_idx, mne.pick_types(info, meg='mag', exclude=[])])
        if 'grad' in config.ch_types:
            pick_idx = np.concatenate(
                [pick_idx, mne.pick_types(info, meg='grad', exclude=[])])
        if 'meg' in config.ch_types:
            pick_idx = mne.pick_types(info, meg=True, eog=True, ecg=True,
                                      exclude=[])
    elif config.ch_types == ['eeg']:
        pick_idx = mne.pick_types(info, meg=False, eeg=True, eog=True,
                                  ecg=True, exclude=[])
    else:
        raise RuntimeError('Something unexpected happened. Please contact '
                           'the mne-bids-pipeline developers. Thank you.')

    ch_names = [info['ch_names'][i] for i in pick_idx]
    return ch_names


def sanitize_cond_name(cond: str) -> str:
    cond = (
        cond
        .replace(os.path.sep, '')
        .replace('_', '')
        .replace('-', '')
        .replace(' ', '')
    )
    return cond


def get_mf_cal_fname(
    *,
    config: SimpleNamespace,
    subject: str,
    session: str
) -> pathlib.Path:
    if config.mf_cal_fname is None:
        mf_cal_fpath = (BIDSPath(subject=subject,
                                 session=session,
                                 suffix='meg',
                                 datatype='meg',
                                 root=get_bids_root(config))
                        .meg_calibration_fpath)
        if mf_cal_fpath is None:
            raise ValueError('Could not find Maxwell Filter Calibration '
                             'file.')
    else:
        mf_cal_fpath = pathlib.Path(
            config.mf_cal_fname
        ).expanduser().absolute()
        if not mf_cal_fpath.exists():
            raise ValueError(f'Could not find Maxwell Filter Calibration '
                             f'file at {str(mf_cal_fpath)}.')

    return mf_cal_fpath


def get_mf_ctc_fname(
    *,
    config: SimpleNamespace,
    subject: str,
    session: str
) -> pathlib.Path:
    if config.mf_ctc_fname is None:
        mf_ctc_fpath = (BIDSPath(subject=subject,
                                 session=session,
                                 suffix='meg',
                                 datatype='meg',
                                 root=get_bids_root(config))
                        .meg_crosstalk_fpath)
        if mf_ctc_fpath is None:
            raise ValueError('Could not find Maxwell Filter cross-talk '
                             'file.')
    else:
        mf_ctc_fpath = pathlib.Path(
            config.mf_ctc_fname).expanduser().absolute()
        if not mf_ctc_fpath.exists():
            raise ValueError(f'Could not find Maxwell Filter cross-talk '
                             f'file at {str(mf_ctc_fpath)}.')

    return mf_ctc_fpath


RawEpochsEvokedT = TypeVar(
    'RawEpochsEvokedT',
    bound=Union[mne.io.BaseRaw, mne.BaseEpochs, mne.Evoked]
)


def _restrict_analyze_channels(
    inst: RawEpochsEvokedT,
    cfg: SimpleNamespace
) -> RawEpochsEvokedT:
    if cfg.analyze_channels:
        analyze_channels = cfg.analyze_channels
        if cfg.analyze_channels == 'ch_types':
            analyze_channels = cfg.ch_types
            inst.apply_proj()
        # We special-case the average reference here to work around a situation
        # where e.g. `analyze_channels` might contain only a single channel:
        # `concatenate_epochs` below will then fail when trying to create /
        # apply the projection. We can avoid this by removing an existing
        # average reference projection here, and applying the average reference
        # directly – without going through a projector.
        elif 'eeg' in cfg.ch_types and cfg.eeg_reference == 'average':
            inst.set_eeg_reference('average')
        else:
            inst.apply_proj()
        inst.pick(analyze_channels)
    return inst


def _get_scalp_in_files(cfg: SimpleNamespace) -> Dict[str, pathlib.Path]:
    subject_path = pathlib.Path(cfg.subjects_dir) / cfg.fs_subject
    seghead = subject_path / 'surf' / 'lh.seghead'
    in_files = dict()
    if seghead.is_file():
        in_files['seghead'] = seghead
    else:
        in_files['t1'] = subject_path / 'mri' / 'T1.mgz'
    return in_files


def _get_bem_conductivity(
    cfg: SimpleNamespace
) -> Tuple[Tuple[float], str]:
    if cfg.fs_subject in ('fsaverage', cfg.use_template_mri):
        conductivity = None  # should never be used
        tag = '5120-5120-5120'
    elif 'eeg' in cfg.ch_types:
        conductivity = (0.3, 0.006, 0.3)
        tag = '5120-5120-5120'
    else:
        conductivity = (0.3,)
        tag = '5120'
    return conductivity, tag


def _meg_in_ch_types(ch_types: str) -> bool:
    return ('mag' in ch_types or 'grad' in ch_types or 'meg' in ch_types)


def get_noise_cov_bids_path(
    cfg: SimpleNamespace,
    subject: str,
    session: Optional[str]
) -> BIDSPath:
    """Retrieve the path to the noise covariance file.

    Parameters
    ----------
    cfg
        The local configuration.
    subject
        The subject identifier.
    session
        The session identifier.

    Returns
    -------
    BIDSPath
        _description_
    """
    noise_cov_bp = BIDSPath(
        subject=subject,
        session=session,
        task=cfg.task,
        acquisition=cfg.acq,
        run=None,
        processing=cfg.proc,
        recording=cfg.rec,
        space=cfg.space,
        suffix='cov',
        extension='.fif',
        datatype=cfg.datatype,
        root=cfg.deriv_root,
        check=False
    )
    noise_cov = cfg.noise_cov
    if callable(noise_cov):
        noise_cov_bp.processing = 'custom'
    elif noise_cov == 'emptyroom':
        noise_cov_bp.task = 'noise'
    elif noise_cov == 'ad-hoc':
        noise_cov_bp.processing = 'adhoc'
    elif noise_cov == 'rest':
        noise_cov_bp.task = 'rest'
    else:  # estimated from a time period
        pass

    return noise_cov_bp


def get_all_contrasts(config: SimpleNamespace) -> Iterable[ArbitraryContrast]:
    _validate_contrasts(config.contrasts)
    normalized_contrasts = []
    for contrast in config.contrasts:
        if isinstance(contrast, tuple):
            normalized_contrasts.append(
                ArbitraryContrast(
                    name=(contrast[0] + "+" + contrast[1]),
                    conditions=list(contrast),
                    weights=[1, -1]
                )
            )
        else:
            normalized_contrasts.append(contrast)
    return normalized_contrasts


def get_decoding_contrasts(
    config: SimpleNamespace
) -> Iterable[Tuple[str, str]]:
    _validate_contrasts(config.contrasts)
    normalized_contrasts = []
    for contrast in config.contrasts:
        if isinstance(contrast, tuple):
            normalized_contrasts.append(contrast)
        else:
            # If a contrast is an `ArbitraryContrast` and satisfies
            # * has exactly two conditions (`check_len`)
            # * weights sum to 0 (`check_sum`)
            # Then the two conditions are used to perform decoding
            check_len = len(contrast["conditions"]) == 2
            check_sum = np.isclose(np.sum(contrast["weights"]), 0)
            if check_len and check_sum:
                cond_1 = contrast["conditions"][0]
                cond_2 = contrast["conditions"][1]
                normalized_contrasts.append((cond_1, cond_2))
    return normalized_contrasts


def get_eeg_reference(
    config: SimpleNamespace
) -> Union[Literal['average'], Iterable[str]]:
    if config.eeg_reference == 'average':
        return config.eeg_reference
    elif isinstance(config.eeg_reference, str):
        return [config.eeg_reference]
    else:
        return config.eeg_reference


def _validate_contrasts(contrasts: SimpleNamespace) -> None:
    for contrast in contrasts:
        if isinstance(contrast, tuple):
            if len(contrast) != 2:
                raise ValueError("Contrasts' tuples MUST be two conditions")
        elif isinstance(contrast, dict):
            if not _keys_arbitrary_contrast.issubset(set(contrast.keys())):
                raise ValueError(f"Missing key(s) in contrast {contrast}")
            if len(contrast["conditions"]) != len(contrast["weights"]):
                raise ValueError(f"Contrast {contrast['name']} has an "
                                 f"inconsistent number of conditions/weights")
        else:
            raise ValueError("Contrasts must be tuples or well-formed dicts")


def _get_script_modules() -> Dict[str, Tuple[ModuleType]]:
    from .scripts import init
    from .scripts import preprocessing
    from .scripts import sensor
    from .scripts import source
    from .scripts import report
    from .scripts import freesurfer

    INIT_SCRIPTS = init.SCRIPTS
    PREPROCESSING_SCRIPTS = preprocessing.SCRIPTS
    SENSOR_SCRIPTS = sensor.SCRIPTS
    SOURCE_SCRIPTS = source.SCRIPTS
    REPORT_SCRIPTS = report.SCRIPTS
    FREESURFER_SCRIPTS = freesurfer.SCRIPTS

    SCRIPT_MODULES = {
        'init': INIT_SCRIPTS,
        'freesurfer': FREESURFER_SCRIPTS,
        'preprocessing': PREPROCESSING_SCRIPTS,
        'sensor': SENSOR_SCRIPTS,
        'source': SOURCE_SCRIPTS,
        'report': REPORT_SCRIPTS,
    }

    # Do not include the FreeSurfer scripts in "all" – we don't intend to run
    # recon-all by default!
    SCRIPT_MODULES['all'] = (
        SCRIPT_MODULES['init'] +
        SCRIPT_MODULES['preprocessing'] +
        SCRIPT_MODULES['sensor'] +
        SCRIPT_MODULES['source'] +
        SCRIPT_MODULES['report']
    )

    return SCRIPT_MODULES