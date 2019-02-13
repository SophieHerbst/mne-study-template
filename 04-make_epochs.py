"""
====================
06. Construct epochs
====================

The epochs are constructed by using the events created in script 03. MNE
supports hierarchical events that allows selection to different groups more
easily (see config.event_id). Automatic rejection is applied to the epochs.
Finally the epochs are saved to disk.
To save space, the epoch data can be decimated.
"""

import os.path as op
import mne
from mne.parallel import parallel_func

import config


# make less parallel runs to limit memory usage
N_JOBS = max(config.N_JOBS // 4, 1)


###############################################################################
# Now we define a function to extract epochs for one subject
def run_epochs(subject):
    print("Processing subject: %s" % subject)

    meg_subject_dir = op.join(config.meg_dir, subject)

    raw_list = list()
    events_list = list()
    print("  Loading raw data")

    for run in config.runs:
        run += '_filt_sss'
        raw_fname = op.join(meg_subject_dir,
                            config.base_raw_fname.format(**locals()))
        eve_fname = op.splitext(raw_fname)[0] + '-eve.fif'

        raw = mne.io.read_raw_fif(raw_fname, preload=True)
        events = mne.read_events(eve_fname)
        events_list.append(events)

        raw.info['bads'] = config.bads[subject]
        raw_list.append(raw)

    raw, events = mne.concatenate_raws(raw_list, events_list=events_list)
    raw.set_eeg_reference(projection=True)
    del raw_list

    picks = mne.pick_types(raw.info, meg=True, eeg=True, stim=True,
                           eog=True, exclude=())

    # Construct metadata from the epochs
    # Add here if you need to attach a pandas dataframe as metadata
    # to your epochs object:
    # https://martinos.org/mne/dev/auto_tutorials/plot_metadata_epochs.html

    # Epoch the data
    print('  Epoching')
    epochs = mne.Epochs(raw, events, config.event_id, config.tmin, config.tmax,
                        proj=True, picks=picks, baseline=config.baseline,
                        preload=False, decim=config.decim,
                        reject=config.reject)

    if config.plot:
        epochs.plot()

    print('  Writing epochs to disk')
    epochs_fname = op.join(meg_subject_dir,
                            config.base_epochs_fname.format(**locals()))

    epochs.save(epochs_fname)
    
    # produce high-pass filtered version of the data for ICA    
    epochs_for_ICA = mne.Epochs(raw.copy().filter(l_freq=1., h_freq=None),
                                events, config.event_id, config.tmin, 
                                config.tmax, proj=True, 
                                picks=picks, baseline=config.baseline,
                                preload=False, decim=config.decim,
                                reject=config.reject)
    
    
    epochs_for_ICA_fname = op.splitext(epochs_fname)[0] + '_for_ICA.fif'

    epochs_for_ICA.save(epochs_for_ICA_fname)

# Here we use fewer N_JOBS to prevent potential memory problems
parallel, run_func, _ = parallel_func(run_epochs, n_jobs=N_JOBS)
parallel(run_func(subject) for subject in config.subjects_list)
