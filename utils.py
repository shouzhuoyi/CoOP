import torch
import torch.nn as nn

from dassl.engine import TRAINER_REGISTRY, TrainerX
#CLIP文件夹完全来自于OPENAI
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

import torch.nn.functional as F
from torch.utils.data import DataLoader
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights,load_checkpoint
import os
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.engine.trainer import TrainerX
from torch.cuda.amp import GradScaler
_tokenizer = _Tokenizer()

#加载预训练权重
def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model



class TextEncoder(nn.Module):
    '''CoOP text'''
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
    
    def forward(self, prompts,token_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2) 
        x = self.ln_final(x)
        x = x[torch.arrange(x.shape[0]), token_prompts, :]@self.text_projection #EOT
        return x
    

class PromptLearning(nn.Module):
    def __init__(self, clip_model,classnames, cfg):
        super().__init__()
        n_cls = len(classnames)
        learnable_token_size = cfg.TRAINER.COOP.N_CTX
        init_token = cfg.TRAINER.COOP.CTX_INIT

        prompt_prefix = "UNDEFINED"
        if init_token != 'random':
            init_token = init_token.replace('_', ' ')
            length = len(init_token.split(' '))
            prompt = clip.tokenize(init_token)
            with torch.no_grad():
                prompt = clip_model.token_embedding(prompt).type(clip_model.dtype)
            
            vector =prompt[0,1:length+1,:]
            prompt_prefix = init_token
            
        else:
            if cfg.Trainer.csc != None: #prompt对于不同类别是否不同学习？
                vectors = torch.empty(n_cls,learnable_token_size, clip_model.ln_final.weight.shape[0],dtype=clip_model.dtype)

            else:
                vectors = torch.empty(learnable_token_size, clip_model.ln_final.weight.shape[0],dtype=clip_model.dtype)   

            
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {learnable_token_size}")

        self.ctx = nn.Parameter(vectors, requires_grad=True)

        classnames = [name.replace('_', ' ') for name in classnames]
        name_len = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + ' ' + name + '.' for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(prompt) for prompt in prompts])
        with torch.no_grad():
            tokenized_prompts = clip_model.token_embedding(tokenized_prompts).type(clip_model.dtype)

        
        self.register_buffer('prefix',tokenized_prompts[:, :1,:])
        self.register_buffer('suffix',tokenized_prompts[:, 1+learnable_token_size:,:])
        self.n_cls = n_cls
        self.learnable_token_size = learnable_token_size
        self.name_len = name_len
        self.tokenized_prompts = tokenized_prompts
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION#PROMPT位置，'end' ,'middle', 'start'



    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == 'end':
            prompts = torch.cat([prefix, ctx, suffix], dim=1)
        elif self.class_token_position == 'middle':
            halfctx = self.learnable_token_size//2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :halfctx, :]
                ctx_i_half2 = ctx[i : i + 1, halfctx:, :]
                prompt = torch.cat([prefix_i, ctx_i_half1, class_i, ctx_i_half2, suffix_i], dim=1)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == 'start':
            prompts = []
            name_len = self.name_lens[i]
            for i in range(self.n_cls):
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        return prompts


class CLIP_WITH_PROMPT(nn.Module):
    def __init__(self, clip_model, classnames, cfg):
        super().__init__()
        self.text_encoder = TextEncoder(clip_model)
        self.prompt_learning = PromptLearning(clip_model, classnames, cfg)
        self.tokenized_prompts = self.prompt_learning.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale


    def forward(self,image):
        image_features = self.image_encoder(image)
        prompts = self.prompt_learning()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits

@TRAINER_REGISTRY.register()
class COOP_TRAINER(TrainerX):
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        self.model = CLIP_WITH_PROMPT(clip_model, classnames, self.cfg)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        
        self.model.to(self.device)

        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

    
    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        output = self.model(image)
        loss = F.cross_entropy(output, label)
        self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = os.path.join(directory, name, model_file)

            if not os.path.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)


        


