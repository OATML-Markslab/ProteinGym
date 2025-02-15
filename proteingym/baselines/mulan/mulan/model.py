from typing import Optional, Union, Tuple

import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss

from transformers import EsmForMaskedLM
from transformers.modeling_outputs import MaskedLMOutput
from transformers.models.esm.modeling_esm import EsmLMHead
from mulan.model_utils import StructEsmModel, NonLinearHead, ContactPredictionHead



class StructEsmForMaskedLM(EsmForMaskedLM):
    _tied_weights_keys = ["lm_head.decoder.weight"]

    def __init__(self, config, num_struct_embeddings_layers=1, struct_data_dim=7, 
                 use_struct_embeddings=True,
                 predict_contacts='none', 
                 predict_angles=False,
                 mask_angle_inputs_with_plddt=True):
        super().__init__(config)

        self.esm = StructEsmModel(config, num_struct_embeddings_layers=num_struct_embeddings_layers, 
                                  struct_data_dim=struct_data_dim, 
                                  use_struct_embeddings=use_struct_embeddings, 
                                  add_pooling_layer=False,
                                  mask_angle_inputs_with_plddt=mask_angle_inputs_with_plddt)
        self.lm_head = EsmLMHead(config)
        self.predict_contacts = predict_contacts
        self.predict_angles = predict_angles
        self.mask_angle_inputs_with_plddt = mask_angle_inputs_with_plddt

        if self.predict_angles:
            self.angle_regression_head = NonLinearHead(
                input_dim=config.hidden_size,
                output_dim=7, # number of angles to predict
                layer_norm_eps=config.layer_norm_eps,
                )

        if self.predict_contacts != 'none':
            self.contact_head = ContactPredictionHead(
                in_features=config.num_hidden_layers * config.num_attention_heads, 
                out_features=1 if self.predict_contacts == 'contact' else 5,
                bias=True,
            )

        self.init_weights()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        struct_inputs: Optional[Union[Tuple, torch.LongTensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[Union[Tuple, torch.LongTensor]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MaskedLMOutput]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        predicted_contacts = None
        if self.predict_contacts != 'none':
            outputs = self.esm(
                input_ids,
                struct_inputs=struct_inputs,
                attention_mask=attention_mask,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=True,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )

            attns = torch.stack(outputs['attentions'], dim=1)  # Matches the original model layout
            # In the original model, attentions for padding tokens are completely zeroed out.
            # This makes no difference most of the time because the other tokens won't attend to them,
            # but it does for the contact prediction task, which takes attentions as input,
            # so we have to mimic that here.
            attns *= attention_mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            attns *= attention_mask.unsqueeze(1).unsqueeze(2).unsqueeze(4)

            predicted_contacts = self.contact_head(input_ids, attentions=attns)

        else:
            outputs = self.esm(
                input_ids,
                struct_inputs=struct_inputs,
                attention_mask=attention_mask,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        sequence_output = outputs[0]

        masked_lm_loss = None
        prediction_angles = None

        prediction_scores = self.lm_head(sequence_output)
        prediction_scores = {'scores': prediction_scores}

        if self.predict_angles:
            prediction_angles = self.angle_regression_head(sequence_output)
            prediction_angles = torch.sigmoid(prediction_angles)
            prediction_scores['angles'] = prediction_angles

        if self.predict_contacts != 'none':
            prediction_scores[self.predict_contacts] = predicted_contacts

        if labels is not None and labels[0] is not None:
            ce_loss_fn = CrossEntropyLoss()
            labels, distance_matrices, angle_labels = labels
            masked_lm_loss = ce_loss_fn(prediction_scores.view(-1, self.config.vocab_size), 
                                        labels.view(-1))
            
            if self.predict_angles:
                angle_mask = angle_labels > -99.
                mse_loss = torch.abs(angle_labels[angle_mask] - prediction_angles[angle_mask])
                mse_loss[mse_loss > 0.5] = 1 - mse_loss[mse_loss > 0.5]
                mse_loss = (mse_loss ** 2).mean()

                mse_weight = 5 #5
                masked_lm_loss += mse_weight * mse_loss
 
            if self.predict_contacts != 'none':
                contact_mask = distance_matrices != -1

                if self.predict_contacts == 'contact':
                    contact_loss_fn = BCEWithLogitsLoss()
                elif self.predict_contacts == 'distance':
                    contact_loss_fn = MSELoss()
                else:
                    contact_loss_fn = CrossEntropyLoss()

                contact_loss = contact_loss_fn(predicted_contacts[contact_mask], 
                                               distance_matrices[contact_mask])

                contact_weight = 0.5 #0.5 
                masked_lm_loss += contact_weight * contact_loss

        ret_attns = None
        if output_attentions:
            ret_attns = outputs.attentions

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=ret_attns,
        )
