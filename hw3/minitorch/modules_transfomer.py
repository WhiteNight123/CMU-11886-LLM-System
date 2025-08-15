import numpy as np
from .tensor import tensor, tensor_from_numpy
from .module import Module, Parameter
from .modules_basic import (
    Embedding,
    Dropout,
    LayerNorm1d,
    Linear
)
from .tensor_ops import TensorBackend
from .nn import (
    max,
    softmax,
    dropout,
    GELU,
)
from typing import Any, Dict, Optional, Sequence, Tuple

datatype = np.float32


class MultiHeadAttention(Module):
    def __init__(self, n_embd: int, n_head: int, causal: bool=True, p_dropout: float=0.1, bias: bool=True, backend: TensorBackend=None):
        super().__init__()
        """Implements Multi-Head Attention as described in "Attention Is All You Need"

        Args:
            n_embd: Dimensionality of embeddings and hidden states
            n_head: Number of heads
            p_dropout: Dropout ratio for dropout layer
            causal: If True, then apply a causal mask during self-attention
            bias: If True, then apply a bias in Linear layers
        
        Attributes:
            q_projection: Linear layer projecting input to Q matrix
            k_projection: Linear layer projecting input to K matrix
            v_projection: Linear layer projecting input to V matrix
            out_projection: Linear output projection layer
            dropout: Dropout layer
        """
        self.backend = backend
        self.n_embd = n_embd 
        self.n_head = n_head
        self.causal = causal
        self.attn_hidden_dim = n_embd // n_head

        ### BEGIN ASSIGN3_3
        # 创建Q、K、V的投影层
        self.q_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.k_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.v_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)
        
        # 创建输出投影层
        self.out_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)
        
        # 创建dropout层
        self.dropout = Dropout(p_dropout=p_dropout)
        ### END ASSIGN3_3

    def create_causal_mask(self, seq_len):
        """
        Create a causal mask for self-attention to prevent information leakage.
        
        Generates a triangular mask where each position can only attend to previous
        positions and itself. Upper triangle contains -inf, lower triangle contains 0.

        Args:
            seq_len (int): Length of the sequence

        Returns:
            Tensor: Causal mask of shape (1, 1, seq_len, seq_len) with -inf above
                    diagonal and 0 on/below diagonal. Will be broadcasted to full
                    attention tensor shape during computation.
        """
        # Returns a 1x1xTxt triangular causal mask for Q @ K^T (You will implicitly broadcast it to BxHxTxT)
        mask = -np.finfo(datatype).max * np.triu(np.ones((1, 1, seq_len, seq_len), dtype=datatype), 1)
        return tensor_from_numpy(mask, backend=self.backend)

    def project_to_query_key_value(self, x):
        """
        Project input embeddings to Query, Key, and Value matrices for self-attention.
        
        Args:
            x (Tensor): Input embeddings of shape (batch_size, seq_len, n_embd)

        Returns:
            tuple: (q, kT, v) where:
                - q: Query matrix of shape (batch_size, num_heads, seq_len, attn_hidden_dim)
                - kT: Transposed key matrix of shape (batch_size, num_heads, attn_hidden_dim, seq_len)
                - v: Value matrix of shape (batch_size, num_heads, seq_len, attn_hidden_dim)
        """
        batch_size, seq_len, n_embd = x.shape
        ### BEGIN ASSIGN3_3
        # 投影到Q、K、V矩阵
        q = self.q_projection(x)  # (batch_size, seq_len, n_embd)
        k = self.k_projection(x)  # (batch_size, seq_len, n_embd)
        v = self.v_projection(x)  # (batch_size, seq_len, n_embd)
        
        # 重塑张量以分离头部
        q = q.contiguous().view(batch_size, seq_len, self.n_head, self.attn_hidden_dim)
        k = k.contiguous().view(batch_size, seq_len, self.n_head, self.attn_hidden_dim)
        v = v.contiguous().view(batch_size, seq_len, self.n_head, self.attn_hidden_dim)
        
        # 转置以获得所需的形状
        q = q.permute(0, 2, 1, 3).contiguous()  # (batch_size, n_head, seq_len, attn_hidden_dim)
        kT = k.permute(0, 2, 3, 1).contiguous()  # (batch_size, n_head, attn_hidden_dim, seq_len)
        v = v.permute(0, 2, 1, 3).contiguous()  # (batch_size, n_head, seq_len, attn_hidden_dim)
        ### END ASSIGN3_3
        return q, kT, v
    
    def self_attention(self, q, kT, v):
        """
        Compute self-attention: softmax((q @ kT) / sqrt(attn_hidden_dim)) @ v.
        
        Args:
            q (Tensor): Query matrix of shape (batch_size, num_heads, seq_len, attn_hidden_dim)
            kT (Tensor): Transposed key matrix of shape (batch_size, num_heads, attn_hidden_dim, seq_len)
            v (Tensor): Value matrix of shape (batch_size, num_heads, seq_len, attn_hidden_dim)

        Returns:
            Tensor: Attention output of shape (batch_size, seq_len, n_embd)
        """
        batch_size, num_head, queries_len, q_dim = q.shape
        _, _, k_dim, _ = kT.shape
        _, _, _, v_dim = v.shape
        assert q_dim == k_dim == v_dim
        result = None
        
        ### BEGIN ASSIGN3_3
        # 计算注意力分数
        attn_scores = q @ kT  # (batch_size, num_head, queries_len, seq_len)
        
        # 缩放注意力分数
        attn_scores = attn_scores / (q_dim ** 0.5)
        
        # 如果是因果注意力，应用掩码
        if self.causal:
            mask = self.create_causal_mask(queries_len)
            attn_scores = attn_scores + mask
        
        # 应用softmax获取注意力权重
        attn_weights = softmax(attn_scores, dim=3)  # 在序列长度维度上应用softmax
        
        # 计算注意力输出
        attn_output = attn_weights @ v  # (batch_size, num_head, queries_len, v_dim)
        
        # 重塑以合并多头注意力
        attn_output = attn_output.permute(0, 2, 1, 3)  # (batch_size, queries_len, num_head, v_dim)
        attn_output = attn_output.contiguous().view(batch_size, queries_len, self.n_embd)
        
        # 通过输出投影层
        result = self.out_projection(attn_output)
        ### END ASSIGN3_3

        return result

    def forward(self, x):
        """
        Compute multi-head attention with optional causal masking.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, n_embd)

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, n_embd)
        """
        batch_size, seq_len, n_embd = x.shape
        ### BEGIN ASSIGN3_3
        # 投影输入到查询、键和值
        q, kT, v = self.project_to_query_key_value(x)
        
        # 计算自注意力
        output = self.self_attention(q, kT, v)
        
        return output
        ### END ASSIGN3_3


class FeedForward(Module):
    def __init__(self, n_embd: int, middle_dim: int=256, p_dropout: float=0.1, bias: bool=True, backend: TensorBackend=None):
        super().__init__()
        """
        Initialize a feed-forward network module.
        
        Args:
            n_embd (int): Input and output dimension
            middle_dim (int): Hidden layer dimension, default 256
            p_dropout (float): Dropout probability, default 0.1
            bias (bool): Whether to use bias in linear layers, default True
            backend (TensorBackend): Backend for tensor operations
            
        Attributes:
            linear_in (Linear): First linear layer
            linear_out (Linear): Second linear layer
            dropout (Dropout): Dropout layer
        """
        ### BEGIN ASSIGN3_3
        self.linear_in  = Linear(n_embd, middle_dim, bias=bias, backend=backend)
        self.linear_out = Linear(middle_dim, n_embd, bias=bias, backend=backend)
        self.dropout    = Dropout(p_dropout)
        ### END ASSIGN3_3

    def forward(self, x):
        """
        Forward pass through feed-forward network with GELU activation and dropout.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, n_embd)

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, n_embd)
        """
        batch_size, seq_len, n_embd = x.shape

        ### BEGIN ASSIGN3_3
        # 通过第一个线性层
        x = GELU(self.linear_in(x.view(batch_size * seq_len, n_embd)))
        x = self.dropout(self.linear_out(x)).view(batch_size, seq_len, n_embd)
        ### END ASSIGN3_3

        return x
    

class TransformerLayer(Module):
    def __init__(self, n_embd: int, n_head: int, p_dropout: float=0.1, ln_eps: float=1e-5, bias: bool=True, backend: TensorBackend=None):
        super().__init__()
        """
        Initialize a transformer layer with pre-layer normalization.
        
        Args:
            n_embd (int): Embedding dimension
            n_head (int): Number of attention heads
            p_dropout (float): Dropout probability, default 0.1
            ln_eps (float): Layer normalization epsilon, default 1e-5
            bias (bool): Whether to use bias in linear layers, default True
            backend (TensorBackend): Backend for tensor operations
            
        Attributes:
            ln_1 (LayerNorm1d): First layer normalization before attention
            ln_2 (LayerNorm1d): Second layer normalization after attention
            attention (MultiHeadAttention): Multi-head attention layer
            ff (FeedForward): Feed-forward network layer
        """
        ### BEGIN ASSIGN3_3
        self.backend = backend
        
        # 创建层归一化层
        self.ln_1 = LayerNorm1d(n_embd, ln_eps, backend=backend)
        self.ln_2 = LayerNorm1d(n_embd, ln_eps, backend=backend)
        
        # 创建多头注意力层
        self.attention = MultiHeadAttention(n_embd, n_head, causal=True, p_dropout=p_dropout, bias=bias, backend=backend)
        
        # 创建前馈网络层
        self.ff = FeedForward(n_embd, middle_dim=4*n_embd, p_dropout=p_dropout, bias=bias, backend=backend)
        ### END ASSIGN3_3

    def forward(self, x):
        """
        Forward pass through transformer layer with pre-layer normalization.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, n_embd)
        
        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, n_embd)
        """
        batch_size, seq_len, n_embd = x.shape
        ### BEGIN YOUR SOLUTION
        # 应用第一个层归一化
        ln_1_out = self.ln_1(x)
        
        # 应用多头注意力并添加残差连接
        attn_out = self.attention(ln_1_out)
        x = x + attn_out
        
        # 应用第二个层归一化
        ln_2_out = self.ln_2(x)
        
        # 应用前馈网络并添加残差连接
        ff_out = self.ff(ln_2_out)
        x = x + ff_out
        
        return x
        ### END YOUR SOLUTION


class DecoderLM(Module):
    def __init__(
        self, 
        n_vocab: int,
        n_embd: int,
        n_head: int,
        n_positions: int,
        p_dropout: float=0.1,
        ln_eps: float=1e-5, 
        bias: bool=True,
        backend: TensorBackend=None
    ):
        super().__init__()
        """
        Initialize a decoder-only transformer language model.
        
        Args:
            n_vocab (int): Vocabulary size
            n_embd (int): Embedding dimension
            n_head (int): Number of attention heads
            n_positions (int): Maximum sequence length
            p_dropout (float): Dropout probability, default 0.1
            ln_eps (float): Layer normalization epsilon, default 1e-5
            bias (bool): Whether to use bias in linear layers, default True
            backend (TensorBackend): Backend for tensor operations
            
        Attributes:
            token_embeddings (Embedding): Token embedding layer
            position_embeddings (Embedding): Position embedding layer
            t_layer_1 (TransformerLayer): First transformer layer
            t_layer_2 (TransformerLayer): Second transformer layer
            t_layer_3 (TransformerLayer): Third transformer layer
            t_layer_4 (TransformerLayer): Fourth transformer layer
            dropout (Dropout): Dropout layer before transformer layers
            ln (LayerNorm1d): Final layer normalization
            lm_head (Linear): Language model head for vocabulary projection
        """
        self.backend = backend
        self.n_embd = n_embd
        self.n_vocab = n_vocab
        ### BEGIN ASSIGN3_3
        # 创建token嵌入层和位置嵌入层
        self.token_embeddings = Embedding(n_vocab, n_embd, backend=backend)
        self.position_embeddings = Embedding(n_positions, n_embd, backend=backend)
        
        # 创建transformer层
        self.t_layer_1 = TransformerLayer(n_embd, n_head, p_dropout, ln_eps, bias, backend=backend)
        self.t_layer_2 = TransformerLayer(n_embd, n_head, p_dropout, ln_eps, bias, backend=backend)
        self.t_layer_3 = TransformerLayer(n_embd, n_head, p_dropout, ln_eps, bias, backend=backend)
        self.t_layer_4 = TransformerLayer(n_embd, n_head, p_dropout, ln_eps, bias, backend=backend)
        
        # 创建dropout层
        self.dropout = Dropout(p_dropout=p_dropout)
        
        # 创建最终的层归一化
        self.ln = LayerNorm1d(n_embd, ln_eps, backend=backend)
        
        # 创建语言模型头部
        self.lm_head = Linear(n_embd, n_vocab, bias=bias, backend=backend)
        ### END ASSIGN3_3
    
    def forward(self, idx):
        """
        Forward pass through decoder-only transformer language model.
        
        Args:
            idx (Tensor): Input token indices of shape (batch_size, seq_len)
        
        Returns:
            Tensor: Logits of shape (batch_size, seq_len, n_vocab)
        """
        
        batch_size, seq_len = idx.shape

        ### BEGIN ASSIGN3_3
        # 1. 获取token嵌入，形状为(batch_size, seq_len, n_embd)
        token_emb = self.token_embeddings(idx)
        
        # 2. 创建位置嵌入，形状为(1, seq_len, n_embd)
        # 创建位置id张量[0, 1, 2, ..., seq_len-1]，形状为(1, seq_len)
        position_ids = tensor_from_numpy(np.arange(seq_len).reshape(1, seq_len).astype(np.float32), backend=self.backend)
        pos_emb = self.position_embeddings(position_ids)
        
        # 3. 将token嵌入和位置嵌入相加
        x = token_emb + pos_emb
        
        # 4. 应用dropout
        x = self.dropout(x)
        
        # 5. 通过transformer层
        x = self.t_layer_1(x)
        x = self.t_layer_2(x)
        x = self.t_layer_3(x)
        x = self.t_layer_4(x)
        
        # 6. 应用最终层归一化
        x = self.ln(x)
        
        # 7. 投影到词汇表大小
        logits = self.lm_head(x)
        
        return logits
        ### END ASSIGN3_3
