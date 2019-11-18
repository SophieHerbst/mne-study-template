"""
===================================
01. Maxwell filter using MNE-Python
===================================

The data are Maxwell filtered using SSS or tSSS (if config.mf_st_duration
is not None) and movement compensation.

Using tSSS with a short duration can be used as an alternative to highpass
filtering. For instance, a duration of 10 s acts like a 0.1 Hz highpass.

The head position of all runs is corrected to the run specified in
config.mf_reference_run.
It is critical to mark bad channels before Maxwell filtering.

The function loads machine-specific calibration files from the paths set for
config.mf_ctc_fname  and config.mf_cal_fname.
"""  # noqa: E501

import os
import os.path as op
import glob
import itertools

from mne.parallel import parallel_func
from mne_bids.read import reader as mne_bids_readers
from mne_bids import make_bids_basename, read_raw_bids
from mne_bids.config import BIDS_VERSION
from mne_bids.utils import _write_json


import config

def run_maxwell_filter(subject, session=None):
    print("Processing subject: %s" % subject)

    print('\nProcessing subject: {}\n{}'
          .format(subject, '-' * (20 + len(subject))))

    # Construct the search path for the data file. `sub` is mandatory
    subject_path = op.join('sub-{}'.format(subject))
    # `session` is optional
    if session is not None:
        subject_path = op.join(subject_path, 'ses-{}'.format(session))

    subject_path = op.join(subject_path, config.kind)
    data_dir = op.join(config.bids_root, subject_path)
    
    for run_idx, run in enumerate(config.runs):
        bids_basename = make_bids_basename(subject=subject,
                                           session=session,
                                           task=config.task,
                                           acquisition=config.acq,
                                           run=run,
                                           processing=config.proc,
                                           recording=config.rec,
                                           space=config.space
                                           )
    
        # Find the data file
        search_str = op.join(data_dir, bids_basename) + '_' + config.kind + '*'
        fnames = sorted(glob.glob(search_str))
        fnames = [f for f in fnames
                  if op.splitext(f)[1] in mne_bids_readers]
    
        if len(fnames) == 1:
            bids_fpath = fnames[0]
        elif len(fnames) == 0:
            raise ValueError('Could not find input data file matching: '
                             '"{}"'.format(search_str))
        elif len(fnames) > 1:
            raise ValueError('Expected to find a single input data file: "{}" '
                             ' but found:\n\n{}'
                             .format(search_str, fnames))
    
        # read_raw_bids automatically
        # - populates bad channels using the BIDS channels.tsv
        # - sets channels types according to BIDS channels.tsv `type` column
        # - sets raw.annotations using the BIDS events.tsv
        _, bids_fname = op.split(bids_fpath)
    
        extra_params = dict(allow_maxshield=True)
        raw = read_raw_bids(bids_fname, config.bids_root, 
                            extra_params=extra_params)
        raw.fix_mag_coil_types()
    
        if config.crop is not None:
            raw.crop(*config.crop)
    
        raw.load_data()
        
        if config.plot:
            # plot raw data
            raw.plot(n_channels=50, butterfly=True)
        
        # Prepare the pipeline directory in /derivatives
        deriv_path = op.join(config.bids_root, 'derivatives', config.PIPELINE_NAME)
        fpath_out = op.join(deriv_path, subject_path)
        if not op.exists(fpath_out):
            os.makedirs(fpath_out)
    
            # Write a dataset_description.json for the pipeline
            ds_json = dict()
            ds_json['Name'] = config.PIPELINE_NAME + ' outputs'
            ds_json['BIDSVersion'] = BIDS_VERSION
            ds_json['PipelineDescription'] = {
                'Name': config.PIPELINE_NAME,
                'Version': config.VERSION,
                'CodeURL': config.CODE_URL,
                }
            ds_json['SourceDatasets'] = {
                'URL': 'n/a',
                }
    
            fname = op.join(deriv_path, 'dataset_description.json')
            _write_json(fname, ds_json, overwrite=True, verbose=True)

        if config.use_maxwell_filter:
            
            if run_idx == 0:
                destination = raw.info['dev_head_t']
    
            if config.mf_st_duration:
                print('    st_duration=%d' % (config.mf_st_duration,))
                
            raw_sss = mne.preprocessing.maxwell_filter(
                raw,
                calibration=config.mf_cal_fname,
                cross_talk=config.mf_ctc_fname,
                st_duration=config.mf_st_duration,
                origin=config.mf_head_origin,
                destination=destination)
            
            # Prepare a name to save the data
            fname_out = op.join(fpath_out, bids_basename + '_sss_raw.fif')
            raw_sss.save(fname_out, overwrite=True)
    
            if config.plot:
                # plot maxfiltered data
                raw_sss.plot(n_channels=50, butterfly=True)
                
        else:
            
            # Prepare a name to save the data
            fname_out = op.join(fpath_out, bids_basename + '_nosss_raw.fif')
            raw.save(fname_out, overwrite=True)
            
            print('Warning: Maxfilter has not been applied.')
            if config.plot:
                # plot maxfiltered data
                raw.plot(n_channels=50, butterfly=True)
            

   
def main():
    """Run maxwell_filter."""
    if not config.use_maxwell_filter:
        return
    parallel, run_func, _ = parallel_func(run_maxwell_filter,
                                          n_jobs=config.N_JOBS)
    parallel(run_func(subject, session) for subject, session in
             itertools.product(config.subjects_list, config.sessions))


if __name__ == '__main__':
    main()
