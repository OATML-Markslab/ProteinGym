from transformers import GPT2Config

class TranceptEVEConfig(GPT2Config):
    """
    Config subclass for Tranception model architecture.
    """
    def __init__(
        self,
        attention_mode="tranception",
        position_embedding="grouped_alibi",
        tokenizer=None,
        full_target_seq=None,
        scoring_window=None,
        inference_time_retrieval_type="TranceptEVE",
        retrieval_aggregation_mode=None, #[substitutions Vs indels]
        retrieval_weights_manual=False,
        retrieval_inference_MSA_weight=0.3,
        retrieval_inference_EVE_weight=0.7,
        MSA_filename=None,
        MSA_weight_file_name=None,
        MSA_start=None,
        MSA_end=None,
        MSA_threshold_sequence_frac_gaps=None,
        MSA_threshold_focus_cols_frac_gaps=None,
        clustal_omega_location=None,
        EVE_model_paths=None,
        EVE_num_samples_log_proba=None,
        EVE_model_parameters_location=None,
        MSA_recalibrate_probas=False,
        EVE_recalibrate_probas=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model_type = "tranception"
        self.attention_mode = attention_mode
        self.position_embedding = position_embedding
        self.tokenizer = tokenizer
        self.full_target_seq = full_target_seq
        self.scoring_window = scoring_window
        self.inference_time_retrieval_type = inference_time_retrieval_type
        self.retrieval_aggregation_mode = retrieval_aggregation_mode
        self.retrieval_weights_manual = retrieval_weights_manual
        self.retrieval_inference_MSA_weight = retrieval_inference_MSA_weight
        self.retrieval_inference_EVE_weight = retrieval_inference_EVE_weight
        self.MSA_filename = MSA_filename
        self.MSA_weight_file_name = MSA_weight_file_name
        self.MSA_start = MSA_start
        self.MSA_end = MSA_end
        self.MSA_threshold_sequence_frac_gaps = MSA_threshold_sequence_frac_gaps
        self.MSA_threshold_focus_cols_frac_gaps = MSA_threshold_focus_cols_frac_gaps
        self.clustal_omega_location = clustal_omega_location
        self.EVE_model_paths = EVE_model_paths
        self.EVE_num_samples_log_proba = EVE_num_samples_log_proba
        self.EVE_model_parameters_location = EVE_model_parameters_location
        self.MSA_recalibrate_probas = MSA_recalibrate_probas
        self.EVE_recalibrate_probas = EVE_recalibrate_probas