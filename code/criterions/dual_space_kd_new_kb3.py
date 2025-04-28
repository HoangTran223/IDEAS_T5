# import math
# import torch
# from .various_divergence import VariousDivergence
# from .ETP_1 import ETP_1
# from .ETP import ETP
# import editdistance
# import cvxpy as cp
# import torch.nn as nn
# import numpy as np
# import torch.nn.functional as F
# import torch.distributed as dist

# class DualSpaceKDWithCMA_OT_3(VariousDivergence):
#     def __init__(self, args, padding_id=-100):
#         super().__init__(args, padding_id=padding_id)
#         print("--------------------Using KB Su dung Multi-OT-KB3-------------------")
#         self.args = args
        
#         if torch.cuda.is_available() and args.precision == "bf16":
#             self.dtype = torch.bfloat16
#         elif torch.cuda.is_available() and args.precision == "fp16":
#             self.dtype = torch.float16
#         else:
#             self.dtype = torch.float32
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         self.window_size = 3
#         self.padding_id = padding_id
#         self.ot_weight_logits = 50.0
#         self.ot_weight_hidden = 50.0
#         self.kd_rate = 0.7
#         self.tau_seq = 2.0
#         self.top_k_vocab = 10
#         self.total_steps = args.total_iters
#         self.current_step = 0
#         self.ngram_size = 3
#         self.sigma = 0.7
#         self._id_mapping_cache = None

#         d_s = args.hidden_dim_student
#         d_t = args.hidden_dim_teacher 
#         self.salience_proj_s = nn.Linear(d_s, 1, bias=True).to(self.device, dtype=self.dtype)

#         # Initialize projection layers
#         self.s2t_projection = nn.Identity()
#         self.lm_head_projection = nn.Identity()

#         self.etp = ETP()
#         self.cost_weights_logits = nn.Parameter(torch.tensor([0.45, 0.25, 0.3], dtype=self.dtype, device=self.device))
#         self.cost_weights_hidden = nn.Parameter(torch.tensor([0.1, 0.2, 0.4, 0.3], dtype=self.dtype, device=self.device))

#     def get_module_output_dim(self, module, input_shape):
#         """Infer the output dimension of a module by passing a dummy input."""
#         with torch.no_grad():
#             dummy_input = torch.zeros(input_shape, device=self.device, dtype=self.dtype)
#             output = module(dummy_input)
#             return output.shape[-1]

#     def forward(
#         self, 
#         distiller, 
#         input_data, 
#         output_data, 
#         logging_output, 
#         batch_denom, 
#     ):
#         self.current_step += 1
#         model = distiller.student_model
#         teacher_model = distiller.teacher_model
#         self.distiller = distiller
#         self.distiller.input_data = input_data

#         with torch.no_grad():
#             teacher_model.eval()
#             teacher_outputs = teacher_model(
#                 input_data[f"teacher_{distiller.teacher_model_type}_input_ids"],
#                 attention_mask=input_data[f"teacher_{distiller.teacher_model_type}_attention_mask"],
#                 position_ids=input_data.get(f"teacher_{distiller.teacher_model_type}_position_ids", None), 
#                 output_hidden_states=True
#             )
        
#         outputs = model(
#             input_data["input_ids"],
#             attention_mask=input_data["attention_mask"],
#             position_ids=input_data.get("position_ids", None), 
#             output_hidden_states=True
#         )

#         logits = outputs.logits.to(self.dtype)
#         log = {}

#         loss_ce = self.compute_cross_entropy_loss(logits, output_data["label"], log=log)[0]
#         log["loss_ce"] = loss_ce

#         kd_loss, log = self.compute_dual_space_kd_loss_with_cma(
#             outputs, teacher_outputs, input_data, output_data, distiller, log
#         )
#         log["kd_loss"] = kd_loss

#         hidden_state_student = outputs.hidden_states[-1].to(self.dtype)
#         hidden_state_teacher = teacher_outputs.hidden_states[-1].to(self.dtype)
        
#         pad_mask = input_data["attention_mask"].bool()
#         teacher_pad_mask = input_data[f"teacher_{distiller.teacher_model_type}_attention_mask"].bool()

#         # ot_loss_logits, log = self.compute_ot_logits(distiller, logits, teacher_outputs.logits.to(self.dtype), 
#         #                                 pad_mask, teacher_pad_mask, hidden_state_student, hidden_state_teacher, log)
#         # ot_loss_hidden, log = self.compute_ot_hidden(distiller, hidden_state_student, hidden_state_teacher, 
#         #                                 pad_mask, teacher_pad_mask, log)

#         # total_loss = loss_ce + self.ot_weight_logits * ot_loss_logits + self.ot_weight_hidden * ot_loss_hidden + self.kd_rate * kd_loss
#         total_loss = loss_ce + + self.kd_rate * kd_loss
#         log["loss"] = total_loss
#         # log["ot_loss_logits"] = ot_loss_logits
#         # log["ot_loss_hidden"] = ot_loss_hidden

#         accuracy = self.compute_token_accuracy(
#             logits, output_data["label"], 
#         )
#         log["accuracy"] = accuracy

#         logging_output = self.record_logging_output(
#             logging_output, batch_denom, log
#         )
#         return total_loss / batch_denom, logging_output
        
#     # def compute_ot_logits(self, distiller, student_logits, teacher_logits, student_mask, teacher_mask, student_outputs, teacher_outputs, log):
#     #     batch_size = student_logits.size(0)
#     #     tau = self.tau_seq
#     #     eps = 1e-7

#     #     def normalize(value):
#     #         means = value.mean(dim=-1, keepdim=True)
#     #         stds = value.std(dim=-1, keepdim=True)
#     #         return value / (stds + 0.0001)

#     #     student_logits = normalize(student_logits).to(self.dtype)
#     #     teacher_logits = normalize(teacher_logits[:, :, :student_logits.size(-1)]).to(self.dtype)

#     #     student_probs = F.softmax(student_logits / tau, dim=-1).to(self.dtype)
#     #     teacher_probs = F.softmax(teacher_logits / tau, dim=-1).to(self.dtype)

#     #     min_vocab = min(student_probs.size(-1), teacher_probs.size(-1))
#     #     k = min(min_vocab, self.top_k_vocab)

#     #     student_prob_sums = student_probs.sum(dim=(0, 1))
#     #     teacher_prob_sums = teacher_probs.sum(dim=(0, 1))

#     #     _, student_topk_indices = torch.topk(student_prob_sums, k=k, dim=-1)
#     #     _, teacher_topk_indices = torch.topk(teacher_prob_sums[:min_vocab], k=k, dim=-1)

#     #     selected_indices = student_topk_indices

#     #     student_logits = student_logits[:, :, selected_indices]
#     #     teacher_logits = teacher_logits[:, :, selected_indices]

#     #     t = min(self.current_step / self.total_steps, 1.0)
#     #     interpolated_teacher_logits = (1 - t) * student_logits + t * teacher_logits

#     #     student_probs = F.softmax(student_logits / tau, dim=-1).to(self.dtype)
#     #     interpolated_teacher_probs = F.softmax(interpolated_teacher_logits / tau, dim=-1).to(self.dtype)

#     #     def improved_sort(value):
#     #         sums = value.sum(dim=(0, 1))
#     #         sorted_indices = torch.argsort(sums, descending=True)
#     #         return value[:, :, sorted_indices]

#     #     student_probs = improved_sort(student_probs)
#     #     interpolated_teacher_probs = improved_sort(interpolated_teacher_probs)

#     #     total_loss = 0
#     #     for b in range(batch_size):
#     #         mask_s = student_mask[b].bool()
#     #         mask_t = teacher_mask[b].bool()
#     #         sp = student_probs[b][mask_s]
#     #         tp = interpolated_teacher_probs[b][mask_t]

#     #         C2 = torch.cdist(tp, sp, p=2).to(self.dtype)

#     #         log_ratio = torch.log(tp.unsqueeze(1) / (sp.unsqueeze(0) + eps))
#     #         C4 = (tp.unsqueeze(1) * log_ratio).sum(dim=-1).to(self.dtype)

#     #         student_seq = student_outputs[b][mask_s]
#     #         teacher_seq = distiller.projectors["ot"](teacher_outputs[b])[mask_t]
#     #         sal_s = torch.sigmoid(self.salience_proj_s(student_seq)).squeeze(-1)
#     #         sal_t = torch.sigmoid(self.salience_proj_s(teacher_seq)).squeeze(-1)
#     #         C_salience = torch.abs(sal_t.unsqueeze(1) - sal_s.unsqueeze(0)).to(self.dtype)

#     #         cost_matrices = [C2, C4, C_salience]
#     #         for i, C in enumerate(cost_matrices):
#     #             if C.shape != cost_matrices[0].shape:
#     #                 raise ValueError(f"Cost matrix {i} has shape {C.shape}, expected {cost_matrices[0].shape}")
#     #         weights = self.cost_weights_logits
#     #         log["avg_c2_logits"] = C2.mean().item()
#     #         log["avg_c4_logits"] = C4.mean().item()
#     #         log["avg_c_salience_logits"] = C_salience.mean().item()

#     #         total_cost = sum(w * C for w, C in zip(weights, cost_matrices))
#     #         total_cost = total_cost.to(self.dtype)
#     #         loss_etp, _ = self.etp(total_cost)
#     #         total_loss += loss_etp

#     #     loss = total_loss * self.ot_weight_logits / batch_size
#     #     log["ot_loss_logits"] = loss.item()
#     #     return loss, log

#     # def compute_ot_hidden(self, distiller, student_outputs, teacher_outputs, attention_mask_student, attention_mask_teacher, log):
#     #     teacher_outputs = distiller.projectors["ot"](teacher_outputs).to(self.dtype)
#     #     batch_size = teacher_outputs.size(0)
#     #     total_loss = 0
#     #     eps = 2e-7

#     #     for b in range(batch_size):
#     #         teacher_seq = teacher_outputs[b]
#     #         student_seq = student_outputs[b]

#     #         teacher_seq = teacher_seq[attention_mask_teacher[b].bool()]
#     #         student_seq = student_seq[attention_mask_student[b].bool()]

#     #         M = teacher_seq.size(0)
#     #         N = student_seq.size(0)

#     #         student_ids = self.distiller.input_data["input_ids"][b][attention_mask_student[b].bool()].tolist()
#     #         teacher_ids = self.distiller.input_data[f"teacher_{distiller.teacher_model_type}_input_ids"][b][attention_mask_teacher[b].bool()].tolist()
#     #         stu_tok = distiller.student_tokenizer.convert_ids_to_tokens(student_ids, skip_special_tokens=True)
#     #         tea_tok = distiller.teacher_tokenizers[distiller.teacher_model_type].convert_ids_to_tokens(teacher_ids, skip_special_tokens=True)

#     #         edit_distance_cache = {}
#     #         def safe_edit(a, b):
#     #             key = (a, b)
#     #             if key in edit_distance_cache:
#     #                 return edit_distance_cache[key]
#     #             val = editdistance.eval(a, b)
#     #             edit_distance_cache[key] = val
#     #             return val
            
#     #         C_s2t = torch.zeros((N, M), device=self.device, dtype=self.dtype)
#     #         pairs_s2t = dtw_alignment(stu_tok, tea_tok, dist_fn_edit)
#     #         for i, j in pairs_s2t:
#     #             if i < N and j < M:
#     #                 C_s2t[i, j] = safe_edit(stu_tok[i], tea_tok[j])

#     #         C_t2s = torch.zeros((M, N), device=self.device, dtype=self.dtype)
#     #         pairs_t2s = dtw_alignment(tea_tok, stu_tok, dist_fn_edit)
#     #         for j, i in pairs_t2s:
#     #             if j < M and i < N:
#     #                 C_t2s[j, i] = safe_edit(tea_tok[j], stu_tok[i])

#     #         C2 = (C_s2t.T + C_t2s) / 2

#     #         C3 = torch.abs(student_seq.unsqueeze(0) - teacher_seq.unsqueeze(1)).sum(dim=-1)
#     #         C3 = C3 / (C3.max() + eps)

#     #         def compute_context_reprs(seq, window):
#     #             ctx = torch.zeros_like(seq)
#     #             for i in range(seq.size(0)):
#     #                 start = max(i - window, 0)
#     #                 end = min(i + window + 1, seq.size(0))
#     #                 ctx[i] = seq[start:end].mean(dim=0)
#     #             return ctx
            
#     #         ctx_s = compute_context_reprs(student_seq, self.window_size)
#     #         ctx_t = compute_context_reprs(teacher_seq, self.window_size)
#     #         C5 = torch.cdist(ctx_t, ctx_s, p=2).to(self.dtype)
#     #         C5 = C5 / (C5.max() + eps)

#     #         ctx_s_norm = ctx_s / (torch.norm(ctx_s, dim=-1, keepdim=True) + eps)
#     #         ctx_t_norm = ctx_t / (torch.norm(ctx_t, dim=-1, keepdim=True) + eps)
#     #         cosine_sim = torch.einsum('md,nd->mn', ctx_t_norm, ctx_s_norm)
#     #         C6 = 1 - cosine_sim 

#     #         cost_matrices = [C2, C3, C5, C6]
#     #         for i, C in enumerate(cost_matrices):
#     #             if C.shape != cost_matrices[0].shape:
#     #                 raise ValueError(f"Cost matrix {i} has shape {C.shape}, expected {cost_matrices[0].shape}")
#     #         weights = self.cost_weights_hidden
#     #         log["avg_c2_last"] = C2.mean().item()
#     #         log["avg_c3_last"] = C3.mean().item()
#     #         log["avg_c5_last"] = C5.mean().item()
#     #         log["avg_c6_last"] = C6.mean().item()

#     #         total_cost = sum(w * C for w, C in zip(weights, cost_matrices))
#     #         total_cost = total_cost.to(self.dtype)
#     #         loss_etp, _ = self.etp(total_cost)
#     #         total_loss += loss_etp

#     #     loss = total_loss * self.ot_weight_hidden / batch_size
#     #     log["ot_loss_hidden"] = loss.item()
#     #     return loss, log
    
#     # def update_cost_weights(self, cost_values_logits, cost_values_hidden):
#     #     def to_scalar_list(values):
#     #         if isinstance(values, torch.Tensor):
#     #             values = values.tolist()
#     #         if not isinstance(values, list):
#     #             logger.error(f"Expected list, got {type(values)}: {values}")
#     #             return None
#     #         try:
#     #             result = []
#     #             for x in values:
#     #                 if isinstance(x, (int, float)):
#     #                     result.append(float(x))
#     #                 elif isinstance(x, (list, tuple)):
#     #                     result.append(float(np.mean(x)))
#     #                 else:
#     #                     logger.error(f"Invalid value in list: {type(x)}, value: {x}")
#     #                     return None
#     #             return result
#     #         except (TypeError, ValueError) as e:
#     #             logger.error(f"Error converting to float: {e}, values: {values}")
#     #             return None
        
#     #     cost_values_logits = to_scalar_list(cost_values_logits)
#     #     cost_values_logits = torch.tensor(cost_values_logits, dtype=self.dtype, device=self.device)

#     #     cost_values_hidden = to_scalar_list(cost_values_hidden)
#     #     cost_values_hidden = torch.tensor(cost_values_hidden, dtype=self.dtype, device=self.device)
        
#     #     c_vals_logits = cost_values_logits.detach().cpu().float().numpy()
#     #     n_logits = len(c_vals_logits)
#     #     sigma = self.sigma
        
#     #     alpha_logits = cp.Variable(n_logits)
#     #     objective_logits = cp.Minimize(c_vals_logits @ alpha_logits + sigma * cp.sum_squares(alpha_logits - 1/n_logits))
#     #     constraints_logits = [cp.sum(alpha_logits) == 1, alpha_logits >= 0.01]
        
#     #     problem_logits = cp.Problem(objective_logits, constraints_logits)
#     #     problem_logits.solve(solver=cp.ECOS, verbose=False)
        
#     #     if alpha_logits.value is None:
#     #         print(f"Rank {dist.get_rank()}: CVXPY solver failed for cost_weights_logits. Skipping update.")
#     #     else:
#     #         new_weights_logits = torch.tensor(alpha_logits.value, dtype=self.cost_weights_logits.dtype, device=self.cost_weights_logits.device)
#     #         with torch.no_grad():
#     #             self.cost_weights_logits.copy_(new_weights_logits)
#     #         alpha_str_logits = ", ".join([f"{w:.6f}" for w in new_weights_logits.tolist()])
#     #         print(alpha_str_logits)
        
#     #     c_vals_hidden = cost_values_hidden.detach().cpu().float().numpy()
#     #     n_hidden = len(c_vals_hidden)
#     #     sigma = self.sigma
        
#     #     alpha_hidden = cp.Variable(n_hidden)
#     #     objective_hidden = cp.Minimize(c_vals_hidden @ alpha_hidden + sigma * cp.sum_squares(alpha_hidden - 1/n_hidden))
#     #     constraints_hidden = [cp.sum(alpha_hidden) == 1, alpha_hidden >= 0.01]
        
#     #     problem_hidden = cp.Problem(objective_hidden, constraints_hidden)
#     #     problem_hidden.solve(solver=cp.ECOS, verbose=False)
        
#     #     if alpha_hidden.value is None:
#     #         print(f"Rank {dist.get_rank()}: CVXPY solver failed for cost_weights_hidden. Skipping update.")
#     #     else:
#     #         new_weights_hidden = torch.tensor(alpha_hidden.value, dtype=self.cost_weights_hidden.dtype, device=self.cost_weights_hidden.device)
#     #         with torch.no_grad():
#     #             self.cost_weights_hidden.copy_(new_weights_hidden)
#     #         alpha_str_hidden = ", ".join([f"{w:.6f}" for w in new_weights_hidden.tolist()])
#     #         print(alpha_str_hidden)
    
#     def compute_ngram_overlap_cost(self, stu_tok, tea_tok, student_ids, teacher_ids, n=3):
#         stu_text = self.distiller.student_tokenizer.decode(student_ids, skip_special_tokens=True).lower()
#         tea_text = self.distiller.teacher_tokenizers[self.distiller.teacher_model_type].decode(teacher_ids, skip_special_tokens=True).lower()
        
#         word_tokens_stu = stu_text.split()
#         word_tokens_tea = tea_text.split()
        
#         stu_ngrams = set(tuple(word_tokens_stu[i:i+n]) for i in range(len(word_tokens_stu)-n+1))
#         tea_ngrams = set(tuple(word_tokens_tea[i:i+n]) for i in range(len(word_tokens_tea)-n+1))
#         common_ngrams = stu_ngrams & tea_ngrams
        
#         N = student_ids.size(0)
#         M = teacher_ids.size(0)
#         C_ngram = torch.ones((N, M), device=self.device, dtype=self.dtype)
        
#         stu_word_map = {}
#         tea_word_map = {}
#         stu_idx = 0
#         tea_idx = 0
#         word_idx = 0
#         stu_text_remaining = stu_text.replace('##', '')
#         tea_text_remaining = tea_text.replace('##', '')
        
#         stu_ids_no_pad = student_ids[student_ids != self.padding_id]
#         for word in word_tokens_stu:
#             while stu_idx < len(stu_ids_no_pad) and word in stu_text_remaining:
#                 stu_word_map[stu_idx] = word_idx
#                 stu_text_remaining = stu_text_remaining.replace(word, '', 1)
#                 stu_idx += 1
#             word_idx += 1
        
#         word_idx = 0
#         tea_ids_no_pad = teacher_ids[teacher_ids != self.padding_id]
#         for word in word_tokens_tea:
#             while tea_idx < len(tea_ids_no_pad) and word in tea_text_remaining:
#                 tea_word_map[tea_idx] = word_idx
#                 tea_text_remaining = tea_text_remaining.replace(word, '', 1)
#                 tea_idx += 1
#             word_idx += 1
        
#         for i in range(N):
#             if i >= len(stu_ids_no_pad):
#                 continue
#             for j in range(M):
#                 if j >= len(tea_ids_no_pad):
#                     continue
#                 if i in stu_word_map and j in tea_word_map:
#                     stu_word_idx = stu_word_map[i]
#                     tea_word_idx = tea_word_map[j]
#                     if stu_word_idx < len(word_tokens_stu) and tea_word_idx < len(word_tokens_tea):
#                         overlap_count = 0
#                         for ngram in common_ngrams:
#                             if (word_tokens_stu[stu_word_idx] in ngram and
#                                 word_tokens_tea[tea_word_idx] in ngram):
#                                 overlap_count += 1
#                         C_ngram[i, j] = 1.0 - (overlap_count / max(1, len(common_ngrams)))
        
#         eps = 1e-7
#         C_ngram = (C_ngram - C_ngram.min()) / (C_ngram.max() - C_ngram.min() + eps)
#         return C_ngram.T
    
#     def compute_dual_space_kd_loss_with_cma(
#         self, outputs, teacher_outputs, input_data, output_data, distiller, log
#     ):
#         target = output_data["label"]  # (B, N)
#         teacher_target = output_data[f"teacher_{distiller.teacher_model_type}_label"]  # (B, M)
#         pad_mask = target.ne(self.padding_id)  # (B, N)
#         teacher_pad_mask = teacher_target.ne(self.padding_id)  # (B, M)

#         hiddens = outputs.hidden_states[-1].to(self.dtype)  # (B, N, hidden_dim_s)
#         teacher_hiddens = teacher_outputs.hidden_states[-1].to(self.dtype)  # (B, M, hidden_dim_t)

#         if hasattr(distiller.student_model, "model") and hasattr(distiller.student_model.model, "embed_tokens"):
#             stu_embed_tokens = distiller.student_model.model.embed_tokens
#         elif hasattr(distiller.student_model, "model") and hasattr(distiller.student_model.model, "model") and hasattr(distiller.student_model.model.model, "embed_tokens"):
#             stu_embed_tokens = distiller.student_model.model.model.embed_tokens
#         elif hasattr(distiller.student_model, "transformer") and hasattr(distiller.student_model.transformer, "wte"):
#             stu_embed_tokens = distiller.student_model.transformer.wte
#         else:
#             raise NotImplementedError

#         if hasattr(distiller.teacher_model, "model") and hasattr(distiller.teacher_model.model, "embed_tokens"):
#             tea_embed_tokens = distiller.teacher_model.model.embed_tokens
#         elif hasattr(distiller.teacher_model, "model") and hasattr(distiller.teacher_model.model, "model") and hasattr(distiller.teacher_model.model.model, "embed_tokens"):
#             tea_embed_tokens = distiller.teacher_model.model.model.embed_tokens
#         elif hasattr(distiller.teacher_model, "transformer") and hasattr(distiller.teacher_model.transformer, "wte"):
#             tea_embed_tokens = distiller.teacher_model.transformer.wte
#         else:
#             raise NotImplementedError

#         formal_target = torch.where(pad_mask, target, torch.zeros_like(target))  # (B, N)
#         formal_input = torch.where(pad_mask, input_data["input_ids"], torch.zeros_like(target))  # (B, N)
#         stu_input_embeds = stu_embed_tokens(formal_input).detach().to(self.dtype)  # (B, N, embed_dim)
#         stu_target_embeds = stu_embed_tokens(formal_target).detach().to(self.dtype)  # (B, N, embed_dim)

#         formal_teacher_target = torch.where(teacher_pad_mask, teacher_target, torch.zeros_like(teacher_target))  # (B, M)
#         formal_teacher_input = torch.where(teacher_pad_mask, input_data[f"teacher_{distiller.teacher_model_type}_input_ids"], torch.zeros_like(teacher_target))  # (B, M)
#         tea_input_embeds = tea_embed_tokens(formal_teacher_input).detach().to(self.dtype)  # (B, M, embed_dim)
#         tea_target_embeds = tea_embed_tokens(formal_teacher_target).detach().to(self.dtype)  # (B, M, embed_dim)

#         stu_index_embeds = torch.cat([stu_input_embeds, stu_target_embeds], -1)  # (B, N, 2*embed_dim)
#         tea_index_embeds = torch.cat([tea_input_embeds, tea_target_embeds], -1)  # (B, M, 2*embed_dim)

#         norm_tea_index_embeds = tea_index_embeds / (tea_index_embeds.std() + 1e-7)  # (B, M, 2*embed_dim)
#         norm_tea_target_embeds = tea_target_embeds / (tea_target_embeds.std() + 1e-7)  # (B, M, embed_dim)
#         norm_teacher_hiddens = teacher_hiddens / (teacher_hiddens.std() + 1e-7)  # (B, M, hidden_dim_t)

#         stu_q_hiddens = distiller.projectors["query"](stu_index_embeds).to(self.dtype)  # (B, N, hidden_dim)
#         tea_k_hiddens = norm_tea_index_embeds.to(self.dtype)  # (B, M, 2*embed_dim)
#         # Fix s2t_hiddens dimension
#         # print(f"[DEBUG] distiller.projectors['s2t'] type: {type(distiller.projectors['s2t'])}, modules: {list(distiller.projectors['s2t'].modules())}")
#         s2t_output_dim = self.get_module_output_dim(distiller.projectors["s2t"], hiddens.shape)
#         hidden_dim_teacher = 4096  # Hardcode to 4096
#         # print(f"[DEBUG] args.hidden_dim_teacher: {self.args.hidden_dim_teacher}, using hidden_dim_teacher: {hidden_dim_teacher}")
#         # print(f"[DEBUG] distiller.projectors['s2t'] output dim: {s2t_output_dim}, expected: {hidden_dim_teacher}")
#         self.s2t_projection = nn.Linear(s2t_output_dim, hidden_dim_teacher, bias=False).to(self.device, dtype=self.dtype)
#         stu_v_hiddens = distiller.projectors["s2t"](hiddens).to(self.dtype)  # (B, N, s2t_output_dim)
#         # print(f"[DEBUG] stu_v_hiddens.shape before projection: {stu_v_hiddens.shape}")
#         stu_v_hiddens = self.s2t_projection(stu_v_hiddens)  # (B, N, hidden_dim_teacher)
#         # print(f"[DEBUG] stu_v_hiddens.shape after projection: {stu_v_hiddens.shape}")
#         tea_v_hiddens = distiller.projectors["t2s"](norm_teacher_hiddens + norm_tea_target_embeds).to(self.dtype)  # (B, M, hidden_dim)
        
#         # Compute align
#         align = stu_q_hiddens.matmul(tea_k_hiddens.transpose(-1, -2)).to(self.dtype)  # (B, N, M)
#         align = align / math.sqrt(2 * teacher_hiddens.shape[-1])
#         batch_size, N, M = align.shape
#         for b in range(batch_size):
#             stu_tok = distiller.student_tokenizer.convert_ids_to_tokens(input_data["input_ids"][b], skip_special_tokens=True)
#             tea_tok = distiller.teacher_tokenizers[distiller.teacher_model_type].convert_ids_to_tokens(input_data[f"teacher_{distiller.teacher_model_type}_input_ids"][b], skip_special_tokens=True)
#             ngram_cost = self.compute_ngram_overlap_cost(stu_tok, tea_tok, input_data["input_ids"][b], input_data[f"teacher_{distiller.teacher_model_type}_input_ids"][b], n=self.ngram_size)
#             align[b] = align[b] - 0.1 * ngram_cost.T
#         align_mask = pad_mask.float().unsqueeze(-1) * teacher_pad_mask.float().unsqueeze(1)  # (B, N, M)
#         align = align + (1.0 - align_mask) * (-100000)

#         student_logits = outputs.logits.to(self.dtype)  # (B, N, V_s)
#         teacher_logits = teacher_outputs.logits[:, :, :student_logits.size(-1)].to(self.dtype)  # (B, M, V_s)
#         # print(f"[DEBUG] student_logits.shape: {student_logits.shape}, expected vocab: {student_logits.size(-1)}")
#         # print(f"[DEBUG] teacher_logits.shape: {teacher_logits.shape}, expected vocab: {student_logits.size(-1)}")
#         # print(f"[DEBUG] hiddens.shape: {hiddens.shape}, teacher_hiddens.shape: {teacher_hiddens.shape}")
#         def normalize(value):
#             means = value.mean(dim=-1, keepdim=True)
#             stds = value.std(dim=-1, keepdim=True)
#             return (value - means) / (stds + 1e-7)
#         student_logits_norm = normalize(student_logits).to(self.dtype)  # (B, N, V_s)
#         teacher_logits_norm = normalize(teacher_logits).to(self.dtype)  # (B, M, V_s)
#         tau = self.tau_seq
#         student_probs = F.softmax(student_logits_norm / tau, dim=-1).to(self.dtype)  # (B, N, V_s)
#         teacher_probs = F.softmax(teacher_logits_norm / tau, dim=-1).to(self.dtype)  # (B, M, V_s)
#         k = min(student_probs.size(-1), teacher_probs.size(-1))
#         student_prob_sums = student_probs.sum(dim=(0, 1))  # (V_s,)
#         teacher_prob_sums = teacher_probs.sum(dim=(0, 1))  # (V_s,)
#         _, student_topk_indices = torch.topk(student_prob_sums, k=k, dim=-1)  # (k,)
#         _, teacher_topk_indices = torch.topk(teacher_prob_sums, k=k, dim=-1)  # (k,)
#         student_logits_selected = student_logits_norm[:, :, student_topk_indices]  # (B, N, k)
#         teacher_logits_selected = teacher_logits_norm[:, :, student_topk_indices]  # (B, M, k)
#         t = 0.5 * (1 - math.cos(math.pi * self.current_step / self.total_steps))
#         interpolated_teacher_logits = (1 - t) * student_logits_selected + t * teacher_logits_selected  # (B, N, k)
#         interpolated_teacher_logits_full = torch.zeros_like(student_logits).to(self.dtype)  # (B, N, V_s)
#         interpolated_teacher_logits_full[:, :, student_topk_indices] = interpolated_teacher_logits

#         teacher_logits_selected_s2t = teacher_logits_norm[:, :, teacher_topk_indices]  # (B, M, k)
#         student_logits_padded = torch.zeros_like(teacher_logits_norm).to(self.dtype)  # (B, M, V_s)
#         student_logits_padded[:, :student_logits_norm.size(1), :] = student_logits_norm
#         student_logits_selected_s2t = student_logits_padded[:, :, teacher_topk_indices]  # (B, M, k)
#         interpolated_teacher_logits_s2t = (1 - t) * student_logits_selected_s2t + t * teacher_logits_selected_s2t  # (B, M, k)
#         interpolated_teacher_logits_s2t_full = torch.zeros_like(teacher_logits).to(self.dtype)  # (B, M, V_s)
#         interpolated_teacher_logits_s2t_full[:, :, teacher_topk_indices] = interpolated_teacher_logits_s2t
#         assert interpolated_teacher_logits_s2t_full.size(-1) == student_logits.size(-1), f"interpolated_teacher_logits_s2t_full vocab size {interpolated_teacher_logits_s2t_full.size(-1)} != student_logits {student_logits.size(-1)}"

#         t2s_weight = torch.softmax(align, -1).to(self.dtype)  # (B, N, M)
#         t2s_hiddens = t2s_weight.matmul(tea_v_hiddens).to(self.dtype)  # (B, N, hidden_dim)
#         t2s_logits = t2s_hiddens.matmul(distiller.student_model.lm_head.weight.detach().transpose(-1, -2).to(self.dtype))  # (B, N, V_s)
#         sal_s = torch.sigmoid(self.salience_proj_s(hiddens)).squeeze(-1)  # (B, N)
#         t2s_ce_loss = self.compute_cross_entropy_loss(t2s_logits, target)[0]
#         t2s_ce_loss_interpolated = self.compute_cross_entropy_loss(interpolated_teacher_logits_full, target)[0]
#         t2s_ce_loss = (0.7 * t2s_ce_loss + 0.3 * t2s_ce_loss_interpolated) * sal_s * pad_mask
#         t2s_ce_loss = t2s_ce_loss.sum() / (sal_s * pad_mask).sum()

#         t2s_kd_loss = self.dist_func(
#             outputs.logits.to(self.dtype), interpolated_teacher_logits_full.detach(), target, reduction="none", use_tea_temp=True
#         )
#         t2s_kd_loss = (t2s_kd_loss * pad_mask).sum() / pad_mask.sum()

#         s2t_weight = torch.softmax(align.transpose(-1, -2), -1).to(self.dtype)  # (B, M, N)
#         for b in range(batch_size):
#             stu_tok = distiller.student_tokenizer.convert_ids_to_tokens(input_data["input_ids"][b], skip_special_tokens=True)
#             tea_tok = distiller.teacher_tokenizers[distiller.teacher_model_type].convert_ids_to_tokens(input_data[f"teacher_{distiller.teacher_model_type}_input_ids"][b], skip_special_tokens=True)
#             pairs_s2t = dtw_alignment(stu_tok, tea_tok, dist_fn_edit)
#             dtw_weight = torch.zeros_like(s2t_weight[b:b+1]).to(self.dtype)  # (1, M, N)
#             for i, j in pairs_s2t:
#                 if i < len(stu_tok) and j < len(tea_tok):
#                     dtw_weight[:, j, i] = 1.0
#             s2t_weight[b:b+1] = 0.7 * s2t_weight[b:b+1] + 0.3 * dtw_weight
#         # print(f"[DEBUG] s2t_weight.shape: {s2t_weight.shape}, stu_v_hiddens.shape: {stu_v_hiddens.shape}")
#         assert s2t_weight.dim() == 3, f"s2t_weight must be 3D, got shape {s2t_weight.shape}"
#         assert stu_v_hiddens.dim() == 3, f"stu_v_hiddens must be 3D, got shape {stu_v_hiddens.shape}"
#         s2t_hiddens = s2t_weight.matmul(stu_v_hiddens)  # (B, M, hidden_dim_teacher)
#         # print(f"[DEBUG] s2t_hiddens.shape: {s2t_hiddens.shape}")
#         assert s2t_hiddens.dim() == 3, f"s2t_hiddens must be 3D, got shape {s2t_hiddens.shape}"
#         assert s2t_hiddens.size(-1) == hidden_dim_teacher, f"s2t_hiddens dim {s2t_hiddens.size(-1)} != hidden_dim_teacher {hidden_dim_teacher}"
#         # Fix teacher_lm_head
#         # print(f"[DEBUG] Original teacher_lm_head.shape: {distiller.teacher_model.lm_head.weight.shape}")
#         teacher_lm_head_full = distiller.teacher_model.lm_head.weight.detach().to(self.dtype)
#         V_s = student_logits.size(-1)  # 50257
#         hidden_dim_teacher = 4096  # Hardcode to 4096
#         # Ensure teacher_lm_head maps from hidden_dim_teacher to V_s
#         if teacher_lm_head_full.size(0) == hidden_dim_teacher and teacher_lm_head_full.size(1) == V_s:
#             teacher_lm_head = teacher_lm_head_full
#         elif teacher_lm_head_full.size(1) == hidden_dim_teacher and teacher_lm_head_full.size(0) == V_s:
#             # Transpose if dimensions are swapped
#             teacher_lm_head = teacher_lm_head_full.transpose(0, 1)
#         else:
#             # Project or adjust dimensions
#             # print(f"[WARNING] teacher_lm_head shape {teacher_lm_head_full.shape} does not match expected ({hidden_dim_teacher}, {V_s}). Adjusting.")
#             # Determine input and output dimensions
#             vocab_dim, hidden_dim = teacher_lm_head_full.size()
#             if hidden_dim != hidden_dim_teacher:
#                 # Case: (V_teacher, hidden_dim_t'), need to project hidden_dim_t' to hidden_dim_teacher
#                 # print(f"[DEBUG] Projecting teacher_lm_head from {hidden_dim} to {hidden_dim_teacher}")
#                 self.lm_head_projection = nn.Linear(hidden_dim, hidden_dim_teacher, bias=False).to(self.device, dtype=self.dtype)
#                 # Apply projection to each vocabulary vector
#                 teacher_lm_head_full = self.lm_head_projection(teacher_lm_head_full)  # (V_teacher, hidden_dim_teacher)
#             else:
#                 teacher_lm_head_full = teacher_lm_head_full  # (hidden_dim_teacher, V_teacher)
#             # Adjust output dimension to V_s
#             if teacher_lm_head_full.size(0) > V_s:
#                 # Slice to match student vocabulary
#                 # print(f"[DEBUG] Slicing teacher_lm_head from {teacher_lm_head_full.size(0)} to {V_s}")
#                 teacher_lm_head_full = teacher_lm_head_full[:V_s, :]
#             elif teacher_lm_head_full.size(0) < V_s:
#                 # Pad with zeros
#                 # print(f"[WARNING] teacher_lm_head has {teacher_lm_head_full.size(0)} rows, expected {V_s}. Padding with zeros.")
#                 teacher_lm_head_full = torch.zeros((V_s, hidden_dim_teacher), device=self.device, dtype=self.dtype)
#                 teacher_lm_head_full[:teacher_lm_head_full.size(0), :] = teacher_lm_head_full
#             # Transpose to get (hidden_dim_teacher, V_s)
#             teacher_lm_head = teacher_lm_head_full.transpose(0, 1)
#         # print(f"[DEBUG] teacher_lm_head.shape after adjustment: {teacher_lm_head.shape}, expected: ({hidden_dim_teacher}, {V_s})")
#         assert teacher_lm_head.size(0) == hidden_dim_teacher, f"teacher_lm_head input dim {teacher_lm_head.size(0)} != hidden_dim_teacher {hidden_dim_teacher}"
#         assert teacher_lm_head.size(1) == V_s, f"teacher_lm_head vocab size {teacher_lm_head.size(1)} != student_logits {V_s}"
#         s2t_logits = s2t_hiddens.matmul(teacher_lm_head)  # (B, M, hidden_dim_teacher) @ (hidden_dim_teacher, V_s) -> (B, M, V_s)
#         # print(f"[DEBUG] s2t_logits.shape: {s2t_logits.shape}")
#         assert s2t_logits.size(-1) == student_logits.size(-1), f"s2t_logits vocab size {s2t_logits.size(-1)} != student_logits {student_logits.size(-1)}"
#         # print(f"[DEBUG] s2t_logits.shape: {s2t_logits.shape}, interpolated_teacher_logits_s2t_full.shape: {interpolated_teacher_logits_s2t_full.shape}")
#         # print(f"[DEBUG] s2t_logits.dtype: {s2t_logits.dtype}, interpolated_teacher_logits_s2t_full.dtype: {interpolated_teacher_logits_s2t_full.dtype}")
#         s2t_kd_loss = self.compute_forward_kl_divergence(
#             s2t_logits, interpolated_teacher_logits_s2t_full.detach(), teacher_target, reduction="none"
#         )
#         s2t_kd_loss = (s2t_kd_loss * teacher_pad_mask).sum() / teacher_pad_mask.sum()

#         t2s_acc = (t2s_logits.argmax(-1).eq(target) * pad_mask).sum()
#         max_probs = (t2s_logits.softmax(-1).max(-1)[0] * pad_mask).sum()
#         s2t_acc = (s2t_logits.argmax(-1).eq(teacher_target) * teacher_pad_mask).sum() * pad_mask.sum() / teacher_pad_mask.sum()
#         kd_loss = t2s_ce_loss + t2s_kd_loss + 0.5 * s2t_kd_loss

#         log["t2s_ce_loss"] = t2s_ce_loss
#         log["t2s_kd_loss"] = t2s_kd_loss
#         log["s2t_kd_loss"] = s2t_kd_loss
#         log["t2s_acc"] = t2s_acc
#         log["s2t_acc"] = s2t_acc
#         log["max_t2s_prob"] = max_probs
#         log["t_interpolation"] = t
#         log["kd_loss"] = kd_loss
#         return kd_loss, log

# def dist_fn_edit(a, b):
#     return editdistance.eval(a, b)

# def dtw_alignment(series_1, series_2, norm_func=dist_fn_edit):
#     matrix = np.zeros((len(series_1) + 1, len(series_2) + 1))
#     matrix[0, :] = np.inf
#     matrix[:, 0] = np.inf
#     matrix[0, 0] = 0
#     for i, vec1 in enumerate(series_1):
#         for j, vec2 in enumerate(series_2):
#             cost = norm_func(vec1, vec2)
#             matrix[i + 1, j + 1] = cost + min(
#                 matrix[i, j + 1], matrix[i + 1, j], matrix[i, j]
#             )
#     matrix = matrix[1:, 1:]
#     i, j = len(series_1) - 1, len(series_2) - 1
#     aligned = []
#     while i > 0 or j > 0:
#         aligned.append((i, j))
#         options = [
#             matrix[i - 1, j - 1] if i > 0 and j > 0 else np.inf,
#             matrix[i - 1, j] if i > 0 else np.inf,
#             matrix[i, j - 1] if j > 0 else np.inf,
#         ]
#         move = np.argmin(options)
#         if move == 0: i -= 1; j -= 1
#         elif move == 1: i -= 1
#         else: j -= 1
#     aligned.append((0, 0))
#     return aligned

# def pairwise_euclidean_distance(x, y):
#     return torch.cdist(x, y, p=2)

# def pairwise_cosin_distance(a, b, eps=1e-8):
#     a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
#     a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n, dtype=torch.bfloat16))
#     b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n, dtype=torch.bfloat16))
#     sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
#     sim_mt = 1 - sim_mt
#     return sim_mt

# def pairwise_attention_distance(x, y, eps=1e-8):
#     d = x.shape[1]
#     sim_mt = torch.mm(x, y.transpose(0, 1)) / math.sqrt(d)
#     attention_weights = torch.softmax(sim_mt, dim=1)
#     dist_mt = 1.0 - attention_weights
#     return dist_mt

import math
import torch
from .various_divergence import VariousDivergence
from .ETP_1 import ETP_1
from .ETP import ETP
import editdistance
import cvxpy as cp
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist

import math
import torch
from .various_divergence import VariousDivergence
import torch.nn as nn
import torch.nn.functional as F


class DualSpaceKDWithCMA_OT_3(VariousDivergence):
    def __init__(self, args, padding_id=-100) -> None:
        super().__init__(args, padding_id=padding_id)
        self.args = args
        self.dtype = torch.float32
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.padding_id = padding_id
        self.kd_rate = 1.5
        self.teacher_lm_head_adjusted = None
        self.total_steps = args.total_iters
        self.current_step = 0
        self.top_k_vocab = 500

        d_s = args.hidden_dim_student
        self.salience_proj_s = nn.Linear(d_s, 1, bias=True).to(self.device, dtype=self.dtype)

    def adjust_teacher_lm_head(self, distiller, V_s, hidden_dim_teacher):
        if self.teacher_lm_head_adjusted is not None:
            return self.teacher_lm_head_adjusted
        
        teacher_lm_head_full = distiller.teacher_model.lm_head.weight.detach().to(self.dtype)
        vocab_dim, hidden_dim = teacher_lm_head_full.size()
        
        if vocab_dim > V_s:
            teacher_lm_head_full = teacher_lm_head_full[:V_s, :]
        elif vocab_dim < V_s:
            teacher_lm_head_full = torch.cat([
                teacher_lm_head_full,
                torch.zeros((V_s - vocab_dim, hidden_dim), device=self.device, dtype=self.dtype)
            ], dim=0)
        
        teacher_lm_head = teacher_lm_head_full.transpose(0, 1)
        self.teacher_lm_head_adjusted = teacher_lm_head
        return teacher_lm_head

    def forward(
        self, 
        distiller, 
        input_data, 
        output_data, 
        logging_output, 
        batch_denom, 
    ):
        self.current_step += 1
        model = distiller.student_model
        teacher_model = distiller.teacher_model
        self.distiller = distiller
        outputs = model(
            input_data["input_ids"],
            attention_mask=input_data["attention_mask"],
            position_ids=input_data.get("position_ids", None), 
            output_hidden_states=True
        )
        logits = outputs.logits.to(self.dtype)
        log = {}

        loss_ce = self.compute_cross_entropy_loss(
            outputs.logits, output_data["label"], log=log
        )[0]
        log["loss_ce"] = loss_ce

        with torch.no_grad():
            teacher_model.eval()
            teacher_outputs = teacher_model(
                input_data[f"teacher_{distiller.teacher_model_type}_input_ids"],
                attention_mask=input_data[f"teacher_{distiller.teacher_model_type}_attention_mask"],
                position_ids=input_data.get(f"teacher_{distiller.teacher_model_type}_position_ids", None), 
                output_hidden_states=True
            )
        
        with torch.cuda.amp.autocast(enabled=False):
            kd_loss, log = self.compute_dual_space_kd_loss_with_cma(
                outputs, teacher_outputs, input_data, output_data, distiller, log
            )
        log["kd_loss"] = kd_loss

        loss = loss_ce + self.kd_rate * kd_loss
        log["loss"] = loss

        accuracy = self.compute_token_accuracy(
            logits, output_data["label"]
        )
        log["accuracy"] = accuracy

        logging_output = self.record_logging_output(
            logging_output, batch_denom, log
        )
        return loss / batch_denom, logging_output
    
    def compute_dual_space_kd_loss_with_cma(
        self, outputs, teacher_outputs, input_data, output_data, distiller, log
    ):
        target = output_data["label"]
        teacher_target = output_data[f"teacher_{distiller.teacher_model_type}_label"]
        pad_mask = target.ne(self.padding_id)
        teacher_pad_mask = teacher_target.ne(self.padding_id)

        hiddens = outputs.hidden_states[-1].to(self.dtype)
        teacher_hiddens = teacher_outputs.hidden_states[-1].to(self.dtype)

        if hasattr(distiller.student_model, "model") \
            and hasattr(distiller.student_model.model, "embed_tokens"):
            stu_embed_tokens = distiller.student_model.model.embed_tokens
        elif hasattr(distiller.student_model, "model") \
            and hasattr(distiller.student_model.model, "model") \
            and hasattr(distiller.student_model.model.model, "embed_tokens"):
            stu_embed_tokens = distiller.student_model.model.model.embed_tokens
        elif hasattr(distiller.student_model, "transformer") \
            and hasattr(distiller.student_model.transformer, "wte"):
            stu_embed_tokens = distiller.student_model.transformer.wte
        else:
            raise NotImplementedError

        if hasattr(distiller.teacher_model, "model") \
            and hasattr(distiller.teacher_model.model, "embed_tokens"):
            tea_embed_tokens = distiller.teacher_model.model.embed_tokens
        elif hasattr(distiller.teacher_model, "model") \
            and hasattr(distiller.teacher_model.model, "model") \
            and hasattr(distiller.teacher_model.model.model, "embed_tokens"):
            tea_embed_tokens = distiller.teacher_model.model.model.embed_tokens
        elif hasattr(distiller.teacher_model, "transformer") \
            and hasattr(distiller.teacher_model.transformer, "wte"):
            tea_embed_tokens = distiller.teacher_model.transformer.wte
        else:
            raise NotImplementedError

        formal_target = torch.where(pad_mask, target, torch.zeros_like(target))
        formal_input = torch.where(pad_mask, input_data["input_ids"], torch.zeros_like(target))
        stu_input_embeds = stu_embed_tokens(formal_input).detach().to(self.dtype)
        stu_target_embeds = stu_embed_tokens(formal_target).detach().to(self.dtype)

        formal_teacher_target = torch.where(teacher_pad_mask, teacher_target, torch.zeros_like(teacher_target))
        formal_teacher_input = torch.where(teacher_pad_mask, input_data[f"teacher_{distiller.teacher_model_type}_input_ids"], torch.zeros_like(teacher_target))
        tea_input_embeds = tea_embed_tokens(formal_teacher_input).detach().to(self.dtype)
        tea_target_embeds = tea_embed_tokens(formal_teacher_target).detach().to(self.dtype)

        stu_index_embeds = torch.cat([stu_input_embeds, stu_target_embeds], -1)
        tea_index_embeds = torch.cat([tea_input_embeds, tea_target_embeds], -1)

        norm_tea_index_embeds = tea_index_embeds / (tea_index_embeds.std() + 1e-7)
        norm_tea_target_embeds = tea_target_embeds / (tea_target_embeds.std() + 1e-7)
        norm_teacher_hiddens = teacher_hiddens / (teacher_hiddens.std() + 1e-7)

        # Đảm bảo projectors ở float32
        for key in ["query", "s2t", "t2s"]:
            if key in distiller.projectors:
                distiller.projectors[key].to(dtype=self.dtype)
        
        stu_q_hiddens = distiller.projectors["query"](stu_index_embeds)
        tea_k_hiddens = norm_tea_index_embeds
        stu_v_hiddens = distiller.projectors["s2t"](hiddens)
        tea_v_hiddens = distiller.projectors["t2s"](norm_teacher_hiddens + norm_tea_target_embeds)
        
        align = stu_q_hiddens.matmul(tea_k_hiddens.transpose(-1, -2))
        align = align / math.sqrt(2 * teacher_hiddens.shape[-1])
        align_mask = pad_mask.float().unsqueeze(-1) * teacher_pad_mask.float().unsqueeze(1)
        align = align + (1.0 - align_mask) * (-100000)

        t2s_weight = torch.softmax(align, -1)        
        t2s_hiddens = t2s_weight.matmul(tea_v_hiddens)
        t2s_logits = t2s_hiddens.matmul(
            distiller.student_model.lm_head.weight.detach().transpose(-1, -2).to(self.dtype)
        )

        # Interpolation
        student_logits = outputs.logits.to(self.dtype)
        teacher_logits = teacher_outputs.logits[:, :, :student_logits.size(-1)].to(self.dtype)
        
        def normalize(value):
            means = value.mean(dim=-1, keepdim=True)
            stds = value.std(dim=-1, keepdim=True)
            return (value - means) / (stds + 1e-7)
        student_logits_norm = normalize(student_logits)
        teacher_logits_norm = normalize(teacher_logits)
        tau = 2.0
        student_probs = F.softmax(student_logits_norm / tau, dim=-1)
        teacher_probs = F.softmax(teacher_logits_norm / tau, dim=-1)
        k = min(student_probs.size(-1), teacher_probs.size(-1), self.top_k_vocab)
        student_prob_sums = student_probs.sum(dim=(0, 1))
        teacher_prob_sums = teacher_probs.sum(dim=(0, 1))
        _, student_topk_indices = torch.topk(student_prob_sums, k=k, dim=-1)
        _, teacher_topk_indices = torch.topk(teacher_prob_sums, k=k, dim=-1)
        student_logits_selected = student_logits_norm[:, :, student_topk_indices]
        teacher_logits_selected = teacher_logits_norm[:, :, student_topk_indices]
        t = 0.5 * (1 - math.cos(math.pi * self.current_step / self.total_steps))
        interpolated_teacher_logits = (1 - t) * student_logits_selected + t * teacher_logits_selected
        interpolated_teacher_logits_full = torch.zeros_like(student_logits)
        interpolated_teacher_logits_full[:, :, student_topk_indices] = interpolated_teacher_logits

        teacher_logits_selected_s2t = teacher_logits_norm[:, :, teacher_topk_indices]
        student_logits_padded = torch.zeros_like(teacher_logits_norm)
        student_logits_padded[:, :student_logits_norm.size(1), :] = student_logits_norm
        student_logits_selected_s2t = student_logits_padded[:, :, teacher_topk_indices]
        interpolated_teacher_logits_s2t = (1 - t) * student_logits_selected_s2t + t * teacher_logits_selected_s2t
        interpolated_teacher_logits_s2t_full = torch.zeros_like(teacher_logits)
        interpolated_teacher_logits_s2t_full[:, :, teacher_topk_indices] = interpolated_teacher_logits_s2t

        # T2S loss
        sal_s = torch.sigmoid(self.salience_proj_s(hiddens)).squeeze(-1)
        t2s_ce_loss = self.compute_cross_entropy_loss(t2s_logits, target)[0]
        t2s_ce_loss_interpolated = self.compute_cross_entropy_loss(interpolated_teacher_logits_full, target)[0]
        t2s_ce_loss = (0.5 * t2s_ce_loss + 0.5 * t2s_ce_loss_interpolated) * sal_s * pad_mask
        t2s_ce_loss = t2s_ce_loss.sum() / (sal_s * pad_mask).sum()
        t2s_acc_mask = t2s_logits.argmax(-1).eq(target)
        t2s_acc = (t2s_acc_mask * pad_mask).sum()
        max_probs = (t2s_logits.softmax(-1).max(-1)[0] * pad_mask).sum()
        log["t2s_ce_loss"] = t2s_ce_loss
        log["t2s_acc"] = t2s_acc
        log["max_t2s_prob"] = max_probs
        log["t_interpolation"] = float(t)
        
        if not self.args.only_save_projector:
            t2s_kd_loss = self.dist_func(
                outputs.logits, interpolated_teacher_logits_full.detach(), target, reduction="none", use_tea_temp=True
            )
            t2s_kd_loss = (t2s_kd_loss * pad_mask * t2s_acc_mask).sum() / pad_mask.sum()

            s2t_weight = torch.softmax(align.transpose(-1, -2), -1)
            s2t_hiddens = s2t_weight.matmul(stu_v_hiddens)
            teacher_lm_head = self.adjust_teacher_lm_head(
                distiller, 
                V_s=outputs.logits.size(-1), 
                hidden_dim_teacher=teacher_hiddens.size(-1)
            )
            s2t_logits = s2t_hiddens.matmul(teacher_lm_head)

            s2t_kd_loss = self.compute_forward_kl_divergence(
                s2t_logits, interpolated_teacher_logits_s2t_full.detach(), teacher_target, reduction="none"
            )
            s2t_kd_loss = (s2t_kd_loss * teacher_pad_mask).sum() / teacher_pad_mask.sum()
            s2t_acc = (s2t_logits.argmax(-1).eq(teacher_target) * teacher_pad_mask).sum() * pad_mask.sum() / teacher_pad_mask.sum()

            kd_loss = t2s_ce_loss + t2s_kd_loss + s2t_kd_loss
            log["t2s_kd_loss"] = t2s_kd_loss
            log["s2t_kd_loss"] = s2t_kd_loss
            log["s2t_acc"] = s2t_acc
        else:
            kd_loss = t2s_ce_loss

        log["kd_loss"] = kd_loss
        return kd_loss, log
