import math
import numpy as np
import torch
import torch.nn as nn
from axial_positional_embedding import AxialPositionalEmbedding
from linear_attention_transformer import LinearAttentionTransformer

#Adapted from ehoogeboom github repo ...

class SinusoidalPosEmb(nn.Module):

    """
    Time embeddings
    """

    def __init__(
            self,
            dim,
            num_steps,
            rescale_steps=4000
        ):

        super().__init__()

        self.dim = dim
        self.num_steps = float(num_steps)
        self.rescale_steps = float(rescale_steps)


    def forward(
            self,
            x
        ):

        x = x/self.num_steps * self.rescale_steps
        device=x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:,None] * emb[None,:]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb




class LinearAttentionTransformerEmbedding(nn.Module):
    
    def __init__(
            self,
            input_dim,
            output_dim,
            dim,
            depth,
            n_blocks,
            max_seq_len,
            num_timesteps,
            heads=8,
            dim_head=None,
            causal=False,
            reversible=False,
            ff_chunks=1,
            ff_glu=False,
            ff_dropout=0.,
            attn_layer_dropout=0.,
            attn_dropout=0.,
            blindspot_size=1,
            n_local_attn_heads=0,
            local_attn_window_size=128,
            return_embeddings=False,
            recieves_context=False,
            pkm_layers=tuple(),
            pkm_num_keys=128,
            attend_axially=False,
            linformer_settings=None,
            context_linformer_settings=None
    ):
        assert (max_seq_len % local_attn_window_size) == 0, 'max sequence length must be divisible by the window size, to calculate number of kmeans cluster'
        super().__init__()

        self.max_seq_len = max_seq_len
        self.depth = depth
        self.emb_dim = dim
        self.n_blocks = n_blocks
        
        print('Input dimension', input_dim)
        print('Output dimension', output_dim)

        # token embeddings
        self.x_emb_NN = nn.Embedding(input_dim, self.emb_dim)

        # time embeddings
        self.time_pos_emb = SinusoidalPosEmb(self.emb_dim, num_timesteps)
        self.mlp = nn.Sequential(
                nn.Linear(self.emb_dim, self.emb_dim*4),
                nn.Softplus(),
                nn.Linear(self.emb_dim*4, self.emb_dim*n_blocks*depth)
        )

        # token positional embeddings
        self.axial_pos_emb = AxialPositionalEmbedding(
                dim = self.emb_dim,
                axial_shape=(
                         max_seq_len // local_attn_window_size,
                         local_attn_window_size)
        )
            
        self.pos_emb = nn.Embedding(1, self.emb_dim)

        self.transformer_blocks = torch.nn.ModuleList()

        for ii in range(n_blocks):

            self.transformer_blocks.append(torch.nn.ModuleList())

            for jj in range(depth):

                self.transformer_blocks[-1].append(
                        LinearAttentionTransformer(
                            self.emb_dim,
                            1,
                            max_seq_len,
                            heads=heads,
                            dim_head=dim_head,
                            causal=causal,
                            ff_chunks=ff_chunks,
                            ff_glu=ff_glu,
                            ff_dropout=ff_dropout,
                            attn_layer_dropout=attn_layer_dropout,
                            reversible=reversible,
                            blindspot_size=blindspot_size,
                            n_local_attn_heads=n_local_attn_heads,
                            local_attn_window_size=local_attn_window_size,
                            attend_axially=attend_axially,
                            linformer_settings=linformer_settings,
                            context_linformer_settings=context_linformer_settings
                        )
                )

        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(self.emb_dim, output_dim) if not return_embeddings else nn.Identity()
#        self.out = nn.Conv1d(self.emb_dim, output_dim, kernel_size=1,stride=1)       


    def forward(self, x, t, **kwargs):


        t = self.time_pos_emb(t)
        t = self.mlp(t)
    
        time_embed = t.reshape(x.size(0), 1, self.emb_dim, self.n_blocks, self.depth)
        x = self.x_emb_NN(x.long()) # final shape (batch_size, timelength, model_emb_dim)
        x_pos = self.axial_pos_emb(x).type(x.type())
        # x_pos = self.pos_emb( self._create_pos_vec(x=x)).type(x.type())
        x_embed_axial = x + x_pos
        h = torch.zeros_like(x_embed_axial)

        for i, block in enumerate(self.transformer_blocks):

            h = h+x_embed_axial
            for j, transformer in enumerate(block):

                h = transformer(h+time_embed[...,i,j])
        
        h = self.norm(h)
        output = self.out(h)

        return output.permute(0,2,1)

class Rezero(nn.Module):

    def __init__(self):
        super(Rezero, self).__init__()
        self.alpha = torch.nn.Parameter(torch.zeros(size=(1,)))

    def forward(self,x):
        return self.alpha * x


def add_model_args(parser):

    # Flow params
    parser.add_argument('--num_steps', type=int, default=1)
    parser.add_argument('--actnorm', type=eval, default=False)
    parser.add_argument('--perm_channel', type=str, default='none', choices={'conv', 'shuffle', 'none'})
    parser.add_argument('--perm_length', type=str, default='reverse', choices={'reverse', 'none'})

    parser.add_argument('--input_dp_rate', type=float, default=0.0)

    # Transformer params.
    parser.add_argument('--transformer_dim', type=int, default=512)
    parser.add_argument('--transformer_heads', type=int, default=16)
    parser.add_argument('--transformer_depth', type=int, default=16)
    parser.add_argument('--transformer_blocks', type=int, default=1)
    parser.add_argument('--transformer_dropout', type=float, default=0.1)
    parser.add_argument('--transformer_reversible', type=eval, default=False)
    parser.add_argument('--transformer_local_heads', type=int, default=8)
    parser.add_argument('--transformer_local_size', type=int, default=128)

def get_model(args, data_shape, num_classes):

    data_shape = data_shape
    num_classes = num_classes
    input_dp_rate = args.input_dp_rate
    transformer_dim = args.transformer_dim
    transformer_heads = args.transformer_heads
    transformer_depth = args.transformer_depth
    transformer_blocks = args.transformer_blocks
    transformer_local_heads = args.transformer_local_heads
    transformer_local_size = args.transformer_local_size
    transformer_reversible = args.transformer_reversible
    diffusion_steps = args.diffusion_steps

    C, L = num_classes, data_shape[0]*data_shape[1]

    print('Data shape index 0:', L)
    current_shape = (L,)

    class DiffTransformer(nn.Module):

        def __init__(self,):

            super(DiffTransformer, self).__init__()

            self.transformer = LinearAttentionTransformerEmbedding(
                    input_dim=num_classes,
                    output_dim=num_classes,
                    dim=transformer_dim,
                    heads=transformer_heads,
                    depth=transformer_depth,
                    n_blocks=transformer_blocks,
                    max_seq_len=L,
                    num_timesteps=diffusion_steps,
                    causal=False, # no autoregression
                    ff_dropout=0, # dropout for feedforward NN
                    attn_layer_dropout=input_dp_rate, # dropout right after self-att layer
                    attn_dropout=0, # dropout post-attention
                    n_local_attn_heads=transformer_local_heads,
                    # number of local attention heads for (QK)*V attention.
                    # this can be a tuple specifying the exact number of local
                    # attention heads at that depth
                    local_attn_window_size=transformer_local_size,
                    # receptive field of the local attention
                    reversible=transformer_reversible,
                    # use reversible nets, from reformer paper
            )
            
            self.rezero = Rezero()

        def forward(self, x, t):
            x = self.transformer(x,t)
    #       x = x.permute(0,2,1)
 #           x = self.rezero(x)
            return x


    model = DiffTransformer()

    return model




