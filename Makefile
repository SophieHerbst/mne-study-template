PYTHON = python

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  fetch         to fetch the data"
	@echo "  preprocess    run sensor space preprocessing from import to epochs"
	@echo "  check         check if the dependencies are available"
	@echo "  profile       to profile memory consumption"
	@echo "  all           to build fetch data, run anatomy scripts and all the processing scripts with no highpass"
	@echo "  clean         remove intermediate scripts generated by processing"

check:
	$(PYTHON) check_system.py

clean:
	$(PYTHON) clean.py

fetch:
	mkdir -p data/system_calibration_files/ && wget https://osf.io/prnzb/download -cO data/system_calibration_files/ct_sparse_nspn.fif
	mkdir -p data/system_calibration_files/ && wget https://osf.io/hyg8k/download -cO data/system_calibration_files/sss_cal_nspn.dat
	mkdir -p data/MEG/SB01/ && wget https://osf.io/k9bth/download -cO data/MEG/SB01/SB01_Localizer_raw.fif
	mkdir -p data/MEG/SB02/ && wget https://osf.io/4rbpd/download -cO data/MEG/SB02/SB02_Localizer_raw.fif
	mkdir -p data/MEG/SB04/ && wget https://osf.io/3nxyv/download -cO data/MEG/SB04/SB04_Localizer_raw.fif
	mkdir -p data/MEG/SB05/ && wget https://osf.io/57qwd/download -cO data/MEG/SB05/SB05_Localizer_raw.fif
	mkdir -p data/MEG/SB06/ && wget https://osf.io/z7ybc/download -cO data/MEG/SB06/SB06_Localizer_raw.fif
	mkdir -p data/MEG/SB07/ && wget https://osf.io/m6wpf/download -cO data/MEG/SB07/SB07_Localizer_raw.fif
	mkdir -p data/MEG/SB08/ && wget https://osf.io/xrhqu/download -cO data/MEG/SB08/SB08_Localizer_raw.fif
	mkdir -p data/MEG/SB09/ && wget https://osf.io/vkftn/download -cO data/MEG/SB09/SB09_Localizer_raw.fif
	mkdir -p data/MEG/SB10/ && wget https://osf.io/29xtm/download -cO data/MEG/SB10/SB10_Localizer_raw.fif
	mkdir -p data/MEG/SB11/ && wget https://osf.io/2zst8/download -cO data/MEG/SB11/SB11_Localizer_raw.fif
	mkdir -p data/MEG/SB12/ && wget https://osf.io/u7cj2/download -cO data/MEG/SB12/SB12_Localizer_raw.fif

preprocess:
	$(PYTHON) 01-import_and_filter.py
	$(PYTHON) 02-apply_maxwell_filter.py
	$(PYTHON) 03-extract_events.py
	$(PYTHON) 04-make_epochs.py
	$(PYTHON) 05a-run_ica.py
	$(PYTHON) 05b-compute_and_apply_ssp.py
	$(PYTHON) 06a-apply_ica.py
	$(PYTHON) 07-make_evoked.py
	$(PYTHON) 08-group_average_sensors.py
	$(PYTHON) 09-sliding_estimator.py
	$(PYTHON) 10-time_frequency.py
	# $(PYTHON) 99-make_reports.py

all:
	$(MAKE) fetch
	$(MAKE) preprocess
