DOWNLOAD_DIR := ./data

create_lock:
	cd environments/poet && conda-lock --kind explicit -f environment.yml -p linux-64

create_stable_env:
	conda create --name poet --file environments/poet/conda-linux-64.lock
	conda run -n poet pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.1.1/flash_attn-2.1.1+cu118torch2.0cxx11abiFALSE-cp39-cp39-linux_x86_64.whl
	conda run -n poet pip install -e .

create_conda_env: create_stable_env

update_conda_env: create_lock create_conda_env

download_model:
	mkdir -p $(DOWNLOAD_DIR)
	wget -c https://zenodo.org/records/11062361/files/poet.ckpt?download=1 -O $(DOWNLOAD_DIR)/poet.ckpt
	wget -c https://zenodo.org/records/11062361/files/LICENSE?download=1 -O $(DOWNLOAD_DIR)/LICENSE

download_and_extract_msas:
	mkdir -p $(DOWNLOAD_DIR)
	wget -c https://zenodo.org/records/11062361/files/proteingymv1_colabfold2302_msas.tar?download=1 -O $(DOWNLOAD_DIR)/proteingymv1_colabfold2302_msas.tar
	cd $(DOWNLOAD_DIR) && tar -xf proteingymv1_colabfold2302_msas.tar && rm proteingymv1_colabfold2302_msas.tar

download_scores:
	mkdir -p $(DOWNLOAD_DIR)
	wget -c https://zenodo.org/records/11062361/files/proteingymv1_scores.tar.gz?download=1 -O $(DOWNLOAD_DIR)/proteingymv1_scores.tar.gz
	cd $(DOWNLOAD_DIR) && tar -xzf proteingymv1_scores.tar.gz && rm proteingymv1_scores.tar.gz

download: download_model download_and_extract_msas download_scores
