import math
import torch
from .various_divergence import VariousDivergence
from .ETP import ETP
import editdistance
import cvxpy as cp
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist


class DualSpaceKDWithCMA_OT_1(VariousDivergence):
    def __init__(self, args, padding_id=-100) -> None:
        super().__init__(args, padding_id=padding_id)
        print("--------------------Using KB 1-------------------")
        self.args = args
        self.etp = ETP()

        if torch.cuda.is_available() and args.precision == "bf16":
            self.dtype = torch.bfloat16
        elif torch.cuda.is_available() and args.precision == "fp16":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ## Add
        self.window_size = 2  
        self.sigma = 0.5 
        self.temperature = 0.1
        d_s = args.hidden_dim_student
        d_t = args.hidden_dim_teacher 

        self.salience_proj_s = nn.Linear(d_s, 1, bias=True).to(self.device, dtype=self.dtype)
        self.salience_proj_t = nn.Linear(d_t, 1, bias=True).to(self.device, dtype=self.dtype)

        self.cost_weights_last = nn.Parameter(
            torch.tensor([0.22, 0.19, 0.28, 0.17, 0.14], dtype=self.dtype, device=self.device)  # C1, C_hybrid, C3, C4, C5, C6
        ) 
        self.cost_weights_first = nn.Parameter(
            torch.tensor([0.35, 0.65], dtype=self.dtype, device=self.device)  # C1, C2, C5
        )

        self.ot_weight = 50.0
        self._id_mapping_cache = None

    def forward(
        self, 
        distiller, 
        input_data, 
        output_data, 
        logging_output, 
        batch_denom, 
    ):
        model = distiller.student_model
        teacher_model = distiller.teacher_model
        self.distiller = distiller

        # Add
        self.distiller.input_data = input_data

        with torch.no_grad():
            teacher_model.eval()
            teacher_outputs = teacher_model(
                input_data[f"teacher_{distiller.teacher_model_type}_input_ids"],
                attention_mask=input_data[f"teacher_{distiller.teacher_model_type}_attention_mask"],
                position_ids=input_data.get(f"teacher_{distiller.teacher_model_type}_position_ids", None), 
                output_hidden_states=True
            )
        
        outputs = model(
            input_data["input_ids"],
            attention_mask=input_data["attention_mask"],
            position_ids=input_data.get("position_ids", None), 
            output_hidden_states=True
        )

        logits = outputs.logits
        log = {}

        loss = 0.0
        loss_ce = self.compute_cross_entropy_loss(outputs.logits, output_data["label"], log=log)[0]
        log["loss_ce"] = loss_ce
        
        kd_loss, log = self.compute_dual_space_kd_loss_with_cma(
            outputs, teacher_outputs, input_data, output_data, distiller, log
        )
        loss = loss_ce + self.kd_rate * kd_loss
        log["kd_loss"] = kd_loss

        hidden_state_student = outputs.hidden_states[-1]  # (batch_size, seq_len_student, hidden_dim_student)
        hidden_state_student_first = outputs.hidden_states[0]
        hidden_state_teacher = teacher_outputs.hidden_states[-1]  # (batch_size, seq_len_teacher, hidden_dim_teacher)
        hidden_state_teacher_first = teacher_outputs.hidden_states[0]
        
        pad_mask = input_data["attention_mask"]
        teacher_pad_mask = input_data[f"teacher_{distiller.teacher_model_type}_attention_mask"]
        pad_mask = pad_mask.bool()
        teacher_pad_mask = teacher_pad_mask.bool()

        ot_loss_last, log = self.compute_etp_loss(
            distiller, hidden_state_student, hidden_state_teacher, pad_mask, 
            teacher_pad_mask, log, is_last_layer=True
        )
        ot_loss_first, log = self.compute_etp_loss(
            distiller, hidden_state_student_first, hidden_state_teacher_first, 
            pad_mask, teacher_pad_mask, log, is_last_layer=False
        )

        ot_loss = (ot_loss_last + ot_loss_first)
        log["ot_loss_last"] = ot_loss_last 
        log["ot_loss_first"] = ot_loss_first
        log["ot_loss"] = ot_loss

        loss += ot_loss
        log["loss"] = loss

        accuracy = self.compute_token_accuracy(
            logits, output_data["label"], 
        )
        log["accuracy"] = accuracy

        logging_output = self.record_logging_output(
            logging_output, batch_denom, log
        )
        return loss / batch_denom, logging_output
    

    def compute_etp_loss(
        self, distiller, student_outputs, teacher_outputs, attention_mask_student, attention_mask_teacher, log, logits=False, is_last_layer=True
    ):
        """
        Compute OT loss between teacher and student outputs
        
        Args:
            teacher_outputs: tensor of shape (batch_size, seq_len1, input_dim)
            student_outputs: tensor of shape (batch_size, seq_len2, output_dim)
            
        Returns:
            loss: scalar tensor
        """
        raw_teacher_outputs = teacher_outputs
        teacher_outputs = distiller.projectors["ot"](teacher_outputs)
        batch_size = teacher_outputs.size(0)
        total_loss = 0
        eps = 1e-7
        # Add

        for b in range(batch_size):
            teacher_seq = teacher_outputs[b]
            teacher_seq_raw = raw_teacher_outputs[b]
            student_seq = student_outputs[b]

            teacher_mask = attention_mask_teacher[b]  # (seq_len1)
            student_mask = attention_mask_student[b]  # (seq_len2)
            
            # Prune sequences based on the mask
            teacher_seq_raw = teacher_seq_raw[teacher_mask.bool()]  # Shape: (valid_seq_len1, hidden_dim_teacher)
            teacher_seq = teacher_seq[teacher_mask.bool()]  # Shape: (valid_seq_len1, hidden_dim = 768)
            student_seq = student_seq[student_mask.bool()]  # Shape: (valid_seq_len2, hidden_dim)
                        
            M = teacher_seq.size(0)  
            N = student_seq.size(0)  
            # print(f"M: {M}, N: {N}")

            # C1 = pairwise_attention_distance(student_seq, teacher_seq)
            # student_norm = student_seq / (torch.norm(student_seq, dim=-1, keepdim=True) + eps)
            # teacher_norm = teacher_seq / (torch.norm(teacher_seq, dim=-1, keepdim=True) + eps)
            # S = student_norm @ teacher_norm.T
            # S = torch.softmax(S / self.temperature, dim=-1)
            # C1 = 1 - S

            # C1 = C1 / (C1.max() + eps)
                        
            student_ids = self.distiller.input_data["input_ids"][b][attention_mask_student[b].bool()].tolist()
            teacher_ids = self.distiller.input_data[f"teacher_{distiller.teacher_model_type}_input_ids"][b][attention_mask_teacher[b].bool()].tolist()

            stu_tok = distiller.student_tokenizer.convert_ids_to_tokens(student_ids, skip_special_tokens=True)
            tea_tok = distiller.teacher_tokenizers[distiller.teacher_model_type].convert_ids_to_tokens(teacher_ids, skip_special_tokens=True)

            edit_distance_cache = {}
            def safe_edit(a, b):
                key = (a, b)
                if key in edit_distance_cache:
                    return edit_distance_cache[key]
                val = editdistance.eval(a, b)
                edit_distance_cache[key] = val
                return val
            
            C_s2t = torch.zeros((N, M), device=student_seq.device)
            pairs_s2t = dtw_alignment(stu_tok, tea_tok, dist_fn_edit)  # student -> teacher
            for i, j in pairs_s2t:
                if i < N and j < M:
                    C_s2t[i, j] = safe_edit(stu_tok[i], tea_tok[j])

            C_t2s = torch.zeros((M, N), device=student_seq.device)
            pairs_t2s = dtw_alignment(tea_tok, stu_tok, dist_fn_edit)  # teacher -> student
            for j, i in pairs_t2s:
                if j < M and i < N:
                    C_t2s[j, i] = safe_edit(tea_tok[j], stu_tok[i])

            C2 = (C_s2t + C_t2s.T) / 2
            C2 = C2 / (C2.max() + eps)
            
            C3 = torch.cdist(student_seq, teacher_seq, p=2)  # (N, M)
            C3 = C3 / (C3.max() + eps)
            
            sp_c4 = torch.softmax(student_seq, dim=-1)  # (N, d_s)
            tp_c4 = torch.softmax(teacher_seq, dim=-1)  # (M, d_s)
            sp_c4_expanded = sp_c4.unsqueeze(1)  # (N, 1, d_s)
            tp_c4_expanded = tp_c4.unsqueeze(0)  # (1, M, d_s)
            C4 = (sp_c4_expanded * (torch.log(sp_c4_expanded + eps) - torch.log(tp_c4_expanded + eps))).sum(dim=-1)  # (N, M)
            C4 = C4 / (C4.max() + eps)

            proj_student_seq = student_seq
            proj_teacher_seq = teacher_seq

            def compute_context_reprs(seq, window):
                ctx = torch.zeros_like(seq)
                for i in range(seq.size(0)):
                    start = max(i - window, 0)
                    end = min(i + window + 1, seq.size(0))
                    ctx[i] = seq[start:end].mean(dim=0)
                return ctx
            
            ctx_s = compute_context_reprs(proj_student_seq, self.window_size)  
            ctx_t = compute_context_reprs(proj_teacher_seq, self.window_size)
            C5 = torch.cdist(ctx_s, ctx_t, p=2)
            C5 = C5 / (C5.max() + eps)

            sal_s = torch.sigmoid(self.salience_proj_s(student_seq)).squeeze(-1)  # (N,)
            sal_t = torch.sigmoid(self.salience_proj_t(teacher_seq_raw)).squeeze(-1)
            C6 = torch.abs(sal_s.unsqueeze(1) - sal_t.unsqueeze(0))  # (N, M)
            C6 = C6 / (C6.max() + eps)

            # print(f"C1: {C1.shape}, C2: {C2.shape}, C3: {C3.shape}, C4: {C4.shape}", "C5: {C5.shape}, C6: {C6.shape}"
            if is_last_layer:
                def compute_ngram_overlap_cost(stu_tok, tea_tok, student_ids, teacher_ids, n=2):
                    # Decode to raw text
                    stu_text = distiller.student_tokenizer.decode(student_ids, skip_special_tokens=True).lower()
                    tea_text = distiller.teacher_tokenizers[distiller.teacher_model_type].decode(teacher_ids, skip_special_tokens=True).lower()
                    
                    # Tokenize to words
                    word_tokens_stu = stu_text.split()
                    word_tokens_tea = tea_text.split()
                    
                    # Compute n-grams
                    stu_ngrams = set(tuple(word_tokens_stu[i:i+n]) for i in range(len(word_tokens_stu)-n+1))
                    tea_ngrams = set(tuple(word_tokens_tea[i:i+n]) for i in range(len(word_tokens_tea)-n+1))
                    common_ngrams = stu_ngrams & tea_ngrams
                    
                    # Create tea2stu_id_mapping dynamically
                    if self._id_mapping_cache is None:
                        tea2stu_id_mapping = {}
                        # Use tea2stu_token_mapping if available
                        if hasattr(distiller, 'tea2stu_token_mapping'):
                            for t_tok, s_tok in distiller.tea2stu_token_mapping.items():
                                t_id = distiller.teacher_tokenizers[distiller.teacher_model_type].convert_tokens_to_ids(t_tok)
                                s_id = distiller.student_tokenizer.convert_tokens_to_ids(s_tok)
                                if t_id is not None and s_id is not None:
                                    tea2stu_id_mapping[str(t_id)] = s_id
                        # Add direct token-to-id mapping
                        for t_id in set(teacher_ids):
                            t_tok = distiller.teacher_tokenizers[distiller.teacher_model_type].convert_ids_to_tokens(t_id)
                            s_id = distiller.student_tokenizer.convert_tokens_to_ids(t_tok)
                            if s_id is not None:
                                tea2stu_id_mapping[str(t_id)] = s_id
                        self._id_mapping_cache = tea2stu_id_mapping
                    
                    # Use cached mapping
                    C_ngram = torch.ones((N, M), device=student_seq.device)
                    for i, s_id in enumerate(student_ids):
                        for j, t_id in enumerate(teacher_ids):
                            t_id_str = str(t_id)
                            if t_id_str in self._id_mapping_cache and self._id_mapping_cache[t_id_str] == s_id:
                                # Check n-gram overlap for aligned tokens
                                stu_text_pos = stu_text
                                tea_text_pos = tea_text
                                for ngram in common_ngrams:
                                    ngram_str = ' '.join(ngram)
                                    if ngram_str in stu_text_pos and ngram_str in tea_text_pos:
                                        C_ngram[i, j] = 0
                                        stu_text_pos = stu_text_pos.replace(ngram_str, '', 1)
                                        tea_text_pos = tea_text_pos.replace(ngram_str, '', 1)
                                        break
                    # Fallback to word-level mapping if no mapping
                    if not self._id_mapping_cache:
                        stu_word_map = {}
                        tea_word_map = {}
                        word_idx = 0
                        stu_idx = 0
                        tea_idx = 0
                        stu_text_remaining = stu_text.replace('##', '')
                        tea_text_remaining = tea_text.replace('##', '')
                        
                        for word in word_tokens_stu:
                            while stu_idx < len(stu_tok) and word in stu_text_remaining:
                                stu_word_map[stu_idx] = word_idx
                                stu_text_remaining = stu_text_remaining.replace(word, '', 1)
                                stu_idx += 1
                            word_idx += 1
                        word_idx = 0
                        for word in word_tokens_tea:
                            while tea_idx < len(tea_tok) and word in tea_text_remaining:
                                tea_word_map[tea_idx] = word_idx
                                tea_text_remaining = tea_text_remaining.replace(word, '', 1)
                                tea_idx += 1
                            word_idx += 1

                        for i in range(N):
                            for j in range(M):
                                if i in stu_word_map and j in tea_word_map:
                                    stu_word_idx = stu_word_map[i]
                                    tea_word_idx = tea_word_map[j]
                                    if stu_word_idx < len(word_tokens_stu) and tea_word_idx < len(word_tokens_tea):
                                        for ngram in common_ngrams:
                                            if (word_tokens_stu[stu_word_idx] in ngram and
                                                word_tokens_tea[tea_word_idx] in ngram):
                                                C_ngram[i, j] = 0
                                                break
                    # Normalize C_ngram
                    C_ngram = C_ngram / (C_ngram.max() + eps)
                    return C_ngram

                C7 = compute_ngram_overlap_cost(stu_tok, tea_tok, student_ids, teacher_ids, n=2)
                beta = 0.7
                C_hybrid = beta * C2 + (1 - beta) * C7

            # Add
            if is_last_layer:
                cost_matrices = [C_hybrid, C3, C4, C5, C6]
                weights = self.cost_weights_last
                cost_values = [C_hybrid.mean(), C3.mean(), C4.mean(), C5.mean(), C6.mean()]
                log["avg_c_hybrid_last"] = C_hybrid.mean().item()
                log["avg_c3_last"] = C3.mean().item()
                log["avg_c4_last"] = C4.mean().item()
                log["avg_c5_last"] = C5.mean().item()
                log["avg_c6_last"] = C6.mean().item()
            else:
                cost_matrices = [C2, C5]
                weights = self.cost_weights_first
                cost_values = [C2.mean(), C5.mean()]
                log["avg_c2_first"] = C2.mean().item()
                log["avg_c5_first"] = C5.mean().item()


            total_cost = sum(w * C for w, C in zip(weights, cost_matrices))
            total_cost = total_cost.to(dtype=self.dtype)
            loss_etp, _ = self.etp(total_cost)
            total_loss += loss_etp

        loss = total_loss * self.ot_weight
        loss = total_loss / batch_size
        return loss, log

   
    def update_cost_weights(self, cost_values_last, cost_values_first):

        def to_scalar_list(values):
            if isinstance(values, torch.Tensor):
                values = values.tolist()
            if not isinstance(values, list):
                logger.error(f"Expected list, got {type(values)}: {values}")
                return None
            try:
                result = []
                for x in values:
                    if isinstance(x, (int, float)):
                        result.append(float(x))
                    elif isinstance(x, (list, tuple)):
                        result.append(float(np.mean(x)))
                    else:
                        logger.error(f"Invalid value in list: {type(x)}, value: {x}")
                        return None
                return result
            except (TypeError, ValueError) as e:
                logger.error(f"Error converting to float: {e}, values: {values}")
                return None
        
        cost_values_last = to_scalar_list(cost_values_last)
        cost_values_first = to_scalar_list(cost_values_first)
        cost_values_last = torch.tensor(cost_values_last, dtype=self.dtype, device=self.device)
        cost_values_first = torch.tensor(cost_values_first, dtype=self.dtype, device=self.device)
        
        ###
        c_vals_last = cost_values_last.detach().cpu().float().numpy()
        n_last = len(c_vals_last)  # 5
        sigma = self.sigma
        
        alpha_last = cp.Variable(n_last)
        objective_last = cp.Minimize(c_vals_last @ alpha_last + sigma * cp.sum_squares(alpha_last - 1/n_last))
        constraints_last = [cp.sum(alpha_last) == 1, alpha_last >= 0.01]
        
        problem_last = cp.Problem(objective_last, constraints_last)
        problem_last.solve(solver=cp.ECOS, verbose=False)
        
        if alpha_last.value is None:
            print(f"Rank {dist.get_rank()}: CVXPY solver failed for cost_weights_last. Skipping update.")
        else:
            new_weights_last = torch.tensor(alpha_last.value, dtype=self.cost_weights_last.dtype, device=self.cost_weights_last.device)
            with torch.no_grad():
                self.cost_weights_last.copy_(new_weights_last)
            alpha_str_last = ", ".join([f"{w:.6f}" for w in new_weights_last.tolist()])
            print(f"Alpha_Last: {alpha_str_last}")
        
        ###
        c_vals_first = cost_values_first.detach().cpu().float().numpy()
        n_first = len(c_vals_first)  # 4
        sigma = self.sigma
        
        alpha_first = cp.Variable(n_first)
        objective_first = cp.Minimize(c_vals_first @ alpha_first + sigma * cp.sum_squares(alpha_first - 1/n_first))
        constraints_first = [cp.sum(alpha_first) == 1, alpha_first >= 0.01]
        
        problem_first = cp.Problem(objective_first, constraints_first)
        problem_first.solve(solver=cp.ECOS, verbose=False)
        
        if alpha_first.value is None:
            print(f"Rank {dist.get_rank()}: CVXPY solver failed for cost_weights_first. Skipping update.")
        else:
            new_weights_first = torch.tensor(alpha_first.value, dtype=self.cost_weights_first.dtype, device=self.cost_weights_first.device)
            with torch.no_grad():
                self.cost_weights_first.copy_(new_weights_first)
            alpha_str_first = ", ".join([f"{w:.6f}" for w in new_weights_first.tolist()])
            print(f"Alpha_First: {alpha_str_first}")



        
    def compute_dual_space_kd_loss_with_cma(
        self, outputs, teacher_outputs, input_data, output_data, distiller, log
    ):
        # Ground truth labels
        target = output_data["label"]

        teacher_target = output_data[f"teacher_{distiller.teacher_model_type}_label"]
        pad_mask = target.ne(self.padding_id)
        teacher_pad_mask = teacher_target.ne(self.padding_id)

        hiddens = outputs.hidden_states[-1]
        teacher_hiddens = teacher_outputs.hidden_states[-1]

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
            and hasattr(distiller.teacher_model.model, "wte"):
            tea_embed_tokens = distiller.teacher_model.transformer.wte
        else:
            raise NotImplementedError

        formal_target = torch.where(pad_mask, target, torch.zeros_like(target))
        formal_input = torch.where(pad_mask, input_data["input_ids"], torch.zeros_like(target))
        stu_input_embeds = stu_embed_tokens(formal_input).detach()
        stu_target_embeds = stu_embed_tokens(formal_target).detach()

        formal_teacher_target = torch.where(teacher_pad_mask, teacher_target, torch.zeros_like(teacher_target))
        formal_teacher_input = torch.where(teacher_pad_mask, input_data[f"teacher_{distiller.teacher_model_type}_input_ids"], torch.zeros_like(teacher_target))
        tea_input_embeds = tea_embed_tokens(formal_teacher_input).detach()
        tea_target_embeds = tea_embed_tokens(formal_teacher_target).detach()

        stu_index_embeds = torch.cat([stu_input_embeds, stu_target_embeds], -1)
        tea_index_embeds = torch.cat([tea_input_embeds, tea_target_embeds], -1)

        norm_tea_index_embeds = tea_index_embeds / tea_index_embeds.std()
        norm_tea_target_embeds = tea_target_embeds / tea_target_embeds.std()
        norm_teacher_hiddens = teacher_hiddens / teacher_hiddens.std()

        ## Add
        stu_q_hiddens = distiller.projectors["query"](stu_index_embeds).float()
        tea_k_hiddens = norm_tea_index_embeds.float()

        ## Add 
        stu_v_hiddens = distiller.projectors["s2t"](hiddens).float()
        tea_v_hiddens = distiller.projectors["t2s"](
            norm_teacher_hiddens + norm_tea_target_embeds
        ).float()
        
        align = stu_q_hiddens.matmul(tea_k_hiddens.transpose(-1, -2))
        align = align / math.sqrt(2 * teacher_hiddens.shape[-1])
        align_mask = pad_mask.float().unsqueeze(-1) * teacher_pad_mask.float().unsqueeze(1)
        align = align + (1.0 - align_mask) * (-100000)

        t2s_weight = torch.softmax(align, -1)        
        t2s_hiddens = t2s_weight.matmul(tea_v_hiddens).to(hiddens)
        t2s_logits = t2s_hiddens.matmul(
            distiller.student_model.lm_head.weight.detach().transpose(-1, -2)
        )

        t2s_ce_loss = self.compute_cross_entropy_loss(t2s_logits, target)[0]
        t2s_acc_mask = t2s_logits.argmax(-1).eq(target)
        t2s_acc = (t2s_acc_mask * pad_mask).sum()
        max_probs = (t2s_logits.softmax(-1).max(-1)[0] * pad_mask).sum()
        log["t2s_ce_loss"] = t2s_ce_loss
        log["t2s_acc"] = t2s_acc
        log["max_t2s_prob"] = max_probs
        
        if not self.args.only_save_projector:  # skip if only train projectors (pre-train projectors)
            t2s_kd_loss = self.dist_func(
                outputs.logits, t2s_logits.detach(), target, reduction="none", use_tea_temp=True
            )
            t2s_kd_loss = (t2s_kd_loss * pad_mask * t2s_acc_mask).sum()

            s2t_weight = torch.softmax(align.transpose(-1, -2), -1)
            s2t_hiddens = s2t_weight.matmul(stu_v_hiddens).to(hiddens)
            s2t_logits = s2t_hiddens.matmul(
            distiller.teacher_model.lm_head.weight.detach().transpose(-1, -2)
            )

            s2t_kd_loss = self.compute_forward_kl_divergence(
                s2t_logits, teacher_outputs.logits, teacher_target, reduction="none"
            )
            s2t_kd_loss = (s2t_kd_loss * teacher_pad_mask).sum()
            s2t_acc = (s2t_logits.argmax(-1).eq(teacher_target) * teacher_pad_mask).sum() * pad_mask.sum() / teacher_pad_mask.sum()

            kd_loss = t2s_ce_loss + t2s_kd_loss + s2t_kd_loss
            log["t2s_kd_loss"] = t2s_kd_loss
            log["s2t_kd_loss"] = s2t_kd_loss
            log["s2t_acc"] = s2t_acc
        else:
            kd_loss = t2s_ce_loss

        log["kd_loss"] = kd_loss
        return kd_loss, log



def dist_fn_edit(a, b):
    return editdistance.eval(a, b)

def dtw_alignment(series_1, series_2, norm_func=dist_fn_edit):
    """Simple DTW based on FUSELLM"""
    matrix = np.zeros((len(series_1) + 1, len(series_2) + 1))
    matrix[0, :] = np.inf
    matrix[:, 0] = np.inf
    matrix[0, 0] = 0
    for i, vec1 in enumerate(series_1):
        for j, vec2 in enumerate(series_2):
            cost = norm_func(vec1, vec2)
            matrix[i + 1, j + 1] = cost + min(
                matrix[i, j + 1], matrix[i + 1, j], matrix[i, j]
            )
    matrix = matrix[1:, 1:]
    i, j = len(series_1) - 1, len(series_2) - 1
    aligned = []
    while i > 0 or j > 0:
        aligned.append((i, j))
        options = [
            matrix[i - 1, j - 1] if i > 0 and j > 0 else np.inf,
            matrix[i - 1, j] if i > 0 else np.inf,
            matrix[i, j - 1] if j > 0 else np.inf,
        ]
        move = np.argmin(options)
        if move == 0: i -= 1; j -= 1
        elif move == 1: i -= 1
        else: j -= 1
    aligned.append((0, 0))
    return aligned

def pairwise_euclidean_distance(x, y):
    return torch.cdist(x, y, p=2)  # Computes pairwise Euclidean distance

def pairwise_cosin_distance(a, b, eps=1e-8):
    # a = a.float()
    # b = b.float()
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n, dtype=torch.bfloat16))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n, dtype=torch.bfloat16))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    
    sim_mt = 1 - sim_mt
    return sim_mt

def pairwise_attention_distance(x, y, eps=1e-8):
    # x = x.float()
    # y = y.float()
    
    d = x.shape[1]
   
    sim_mt = torch.mm(x, y.transpose(0, 1)) / math.sqrt(d)
    attention_weights = torch.softmax(sim_mt, dim=1)

    dist_mt = 1.0 - attention_weights
    return dist_mt


