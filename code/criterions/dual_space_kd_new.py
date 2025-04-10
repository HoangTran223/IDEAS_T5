import math
import torch
from .various_divergence import VariousDivergence
from .ETP import ETP
import editdistance

class DualSpaceKDWithCMA_OT(VariousDivergence):
    def __init__(self, args, padding_id=-100) -> None:
        super().__init__(args, padding_id=padding_id)
        self.args = args
        self.etp = ETP()
        self.proj_U = nn.Linear(args.hidden_dim_student, args.max_teacher_len, bias=False)
        self.proj_V = nn.Linear(args.hidden_dim_teacher, args.max_student_len, bias=False)
        
        self.window_size = 2  # có thể để hyperparam
        self.salience_proj_s = nn.Linear(args.hidden_dim_student, 1, bias=True)
        self.salience_proj_t = nn.Linear(args.hidden_dim_teacher, 1, bias=True)
        self.cost_weights = nn.Parameter(torch.tensor(
            [0.15, 0.15, 0.2, 0.2, 0.15, 0.15], dtype=torch.float32
        ))


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
        #

        outputs = model(
            input_data["input_ids"],
            attention_mask=input_data["attention_mask"],
            position_ids=input_data.get("position_ids", None), 
            output_hidden_states=True
        )


        with torch.no_grad():
            teacher_model.eval()
            teacher_outputs = teacher_model(
                input_data[f"teacher_{distiller.teacher_model_type}_input_ids"],
                attention_mask=input_data[f"teacher_{distiller.teacher_model_type}_attention_mask"],
                position_ids=input_data.get(f"teacher_{distiller.teacher_model_type}_position_ids", None), 
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

        target = output_data["label"]
        teacher_target = output_data[f"teacher_{distiller.teacher_model_type}_label"]
        pad_mask = target.ne(self.padding_id)
        teacher_pad_mask = teacher_target.ne(self.padding_id)

        ot_loss_last, log = self.compute_etp_loss(distiller, hidden_state_student, hidden_state_teacher, pad_mask, teacher_pad_mask, log)
        ot_loss_first, log = self.compute_etp_loss(distiller, hidden_state_student_first, hidden_state_teacher_first, pad_mask, teacher_pad_mask, log)

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
        self, distiller, student_outputs, teacher_outputs, attention_mask_student, attention_mask_teacher, log, logits = False
    ):
        """
        Compute OT loss between teacher and student outputs
        
        Args:
            teacher_outputs: tensor of shape (batch_size, seq_len1, input_dim)
            student_outputs: tensor of shape (batch_size, seq_len2, output_dim)
            
        Returns:
            loss: scalar tensor
        """
    
        if logits:
            teacher_outputs = distiller.projectors["ot"](teacher_outputs)
        else:
            teacher_outputs = distiller.projectors["ot"](teacher_outputs)

        batch_size = teacher_outputs.size(0)
        total_loss = 0
        eps = 1e-8

        U = self.proj_U  # nn.Linear(d_student, M)
        V = self.proj_V  # nn.Linear(d_teacher, N)

        for b in range(batch_size):
            # Get sequences for current batch
            teacher_seq = teacher_outputs[b]
            student_seq = student_outputs[b]
            teacher_mask = attention_mask_teacher[b]  # (seq_len1)
            student_mask = attention_mask_student[b]  # (seq_len2)
            
            # Prune sequences based on the mask
            teacher_seq = teacher_seq[teacher_mask.bool()]  # Shape: (valid_seq_len1, hidden_dim)
            student_seq = student_seq[student_mask.bool()]  # Shape: (valid_seq_len2, hidden_dim)
            
            N, M = student_seq.size(0), teacher_seq.size(0)
            C1 = pairwise_attention_distance(student_seq, teacher_seq)
            
            student_ids = self.distiller.input_data["input_ids"][b][attention_mask_student[b]].tolist()
            teacher_ids = self.distiller.input_data[f"teacher_{distiller.teacher_model_type}_input_ids"][b][attention_mask_teacher[b]].tolist()
            stu_special = TOKENIZER_TO_SPECIAL_TOKEN[distiller.student_tokenizer.__class__]
            tea_special = TOKENIZER_TO_SPECIAL_TOKEN[distiller.teacher_tokenizers[distiller.teacher_model_type].__class__]

            stu_tok = distiller.student_tokenizer.convert_ids_to_tokens(student_ids)
            tea_tok = distiller.teacher_tokenizers[distiller.teacher_model_type].convert_ids_to_tokens(teacher_ids)

            C_s2t = torch.zeros((N, M), device=student_seq.device)
            pairs_s2t = dtw_align(stu_tok, tea_tok)  # student -> teacher
            for i, j in pairs_s2t:
                if i < N and j < M:
                    C_s2t[i, j] = dist_fn(stu_tok[i], tea_tok[j])

            C_t2s = torch.zeros((M, N), device=student_seq.device)
            pairs_t2s = dtw_align(tea_tok, stu_tok)  # teacher -> student
            for j, i in pairs_t2s:
                if j < M and i < N:
                    C_t2s[j, i] = dist_fn(tea_tok[j], stu_tok[i])

            C2 = (C_s2t + C_t2s.T) / 2

            sp = U(student_seq)  # (N, M)
            tp = V(teacher_seq).transpose(0, 1) # (N, M)

            C3 = torch.abs(sp - tp)  
            C4 = - sp * torch.log(tp + eps) 

            def get_context_repr(seq, i, w):
                start = max(i - w, 0)
                end = min(i + w + 1, seq.size(0))
                return seq[start:end].mean(dim=0)

            proj_student_seq = distiller.projectors["ot"](student_seq)  # (N, d)
            proj_teacher_seq = distiller.projectors["ot"](teacher_seq)  # (M, d)

            def get_context_repr(seq, i, w):
                start = max(i - w, 0)
                end = min(i + w + 1, seq.size(0))
                return seq[start:end].mean(dim=0)  # (d,)

            C5 = torch.zeros((N, M), device=student_seq.device)
            for i in range(N):
                s_ctx = get_context_repr(proj_student_seq, i, self.window_size)  # (d,)
                for j in range(M):
                    t_ctx = get_context_repr(proj_teacher_seq, j, self.window_size)  # (d,)
                    C5[i, j] = torch.norm(s_ctx - t_ctx, p=2) 
            
            sal_s = torch.sigmoid(self.salience_proj_s(student_seq)).squeeze(-1)  # (N,)
            sal_t = torch.sigmoid(self.salience_proj_t(teacher_seq)).squeeze(-1)  # (M,)
            C6 = torch.abs(sal_s.unsqueeze(1) - sal_t.unsqueeze(0))  # (N, M)
            
            total_cost = (
                self.cost_weights[0] * C1 +
                self.cost_weights[1] * C2 +
                self.cost_weights[2] * C3 +
                self.cost_weights[3] * C4 +
                self.cost_weights[4] * C5 +
                self.cost_weights[5] * C6
            )

            loss_etp, _ = self.etp.sinkhorn(total_cost)
            total_loss += loss_etp

        loss = total_loss / (batch_size * 4)
        log["ot_loss"] = loss
        log["avg_c1"] = C1.mean().item()
        log["avg_c2"] = C2.mean().item()
        log["avg_c3"] = C3.mean().item()
        log["avg_c4"] = C4.mean().item()
        log["avg_c5"] = C5.mean().item()
        log["avg_c6"] = C6.mean().item()

        return loss, log
    
    def update_cost_weights(self, cost_values):
        """
        Cập nhật trọng số cost theo công thức Lagrange đã giải:
            alpha_i = 1/3 - (c_i - avg(c))/ (2*sigma)
        Sau đó, chuẩn hóa để tổng bằng 1.
        """
        avg_cost = torch.mean(cost_values)
        new_weights = 1/3 - (cost_values - avg_cost) / (2 * self.sigma)
        new_weights = new_weights / new_weights.sum()
        self.cost_weights.data = new_weights

        alpha_str = ", ".join([f"{w:.4f}" for w in new_weights.tolist()])
        print_rank(f"Updated alpha weights: [{alpha_str}]")
        
    
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

        stu_q_hiddens = distiller.projectors["query"](stu_index_embeds).float()
        tea_k_hiddens = norm_tea_index_embeds.float()

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
    

def dist_fn_edit(a, b, stu_special_token, tea_special_token):
    aa = a.replace(stu_special_token, "")
    bb = b.replace(tea_special_token, "")
    return editdistance.eval(aa, bb)

def dtw_alignment(series_1, series_2, norm_func):
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