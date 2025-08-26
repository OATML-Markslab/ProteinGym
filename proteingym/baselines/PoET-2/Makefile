PROTEINGYM_CACHE := $(HOME)/.cache/ProteinGym
POET_1_DOWNLOAD_DIR := $(PROTEINGYM_CACHE)/baselines/PoET
POET_2_DOWNLOAD_DIR := $(PROTEINGYM_CACHE)/baselines/PoET-2

download_proteingym_data:
	mkdir -p $(PROTEINGYM_CACHE)
	cd $(PROTEINGYM_CACHE) && \
	VERSION="v1.3" && \
	FILENAME=$(filename).zip && \
	curl -o $${FILENAME} https://marks.hms.harvard.edu/proteingym/ProteinGym_$${VERSION}/$${FILENAME} && \
	unzip $${FILENAME} && \
	rm $${FILENAME}

download_zero_shot_data:
	$(MAKE) download_proteingym_data filename=DMS_ProteinGym_substitutions
	$(MAKE) download_proteingym_data filename=DMS_ProteinGym_indels
	$(MAKE) download_proteingym_data PROTEINGYM_CACHE=$(PROTEINGYM_CACHE)/zero_shot_substitutions_scores filename=zero_shot_substitutions_scores
	$(MAKE) download_proteingym_data filename=zero_shot_indels_scores

download_supervised_data:
	$(MAKE) download_proteingym_data filename=cv_folds_singles_substitutions
	$(MAKE) download_proteingym_data filename=cv_folds_multiples_substitutions
	$(MAKE) download_proteingym_data filename=cv_folds_indels
	$(MAKE) download_proteingym_data filename=DMS_supervised_substitutions_scores
	$(MAKE) download_proteingym_data filename=DMS_supervised_indels_scores

download_clinical:
	$(MAKE) download_proteingym_data PROTEINGYM_CACHE=$(PROTEINGYM_CACHE)/clinical_ProteinGym_substitutions filename=clinical_ProteinGym_substitutions
	$(MAKE) download_proteingym_data PROTEINGYM_CACHE=$(PROTEINGYM_CACHE)/clinical_ProteinGym_indels filename=clinical_ProteinGym_indels
	$(MAKE) download_proteingym_data PROTEINGYM_CACHE=$(PROTEINGYM_CACHE)/zero_shot_clinical_substitutions_scores filename=zero_shot_clinical_substitutions_scores
	$(MAKE) download_proteingym_data PROTEINGYM_CACHE=$(PROTEINGYM_CACHE)/zero_shot_clinical_indels_scores filename=zero_shot_clinical_indels_scores

download_proteingym_af2:
	$(MAKE) download_proteingym_data filename=ProteinGym_AF2_structures

download_proteingym: download_zero_shot_data download_supervised_data download_clinical download_proteingym_af2


download_model:
	mkdir -p $(POET_2_DOWNLOAD_DIR)
	wget -c https://zenodo.org/records/16815645/files/poet-2.ckpt?download=1 -O $(POET_2_DOWNLOAD_DIR)/poet-2.ckpt
	wget -c https://zenodo.org/records/16815645/files/MODEL_LICENSE?download=1 -O $(POET_2_DOWNLOAD_DIR)/MODEL_LICENSE

download_and_extract_msas:
	mkdir -p $(POET_1_DOWNLOAD_DIR)
	wget -c https://zenodo.org/records/11062361/files/proteingymv1_colabfold2302_msas.tar?download=1 -O $(POET_1_DOWNLOAD_DIR)/proteingymv1_colabfold2302_msas.tar
	cd $(POET_1_DOWNLOAD_DIR) && tar -xf proteingymv1_colabfold2302_msas.tar && rm proteingymv1_colabfold2302_msas.tar

download: download_model download_and_extract_msas
