# Assignment 4: Transformer CUDA Acceleration

In this assignment, you'll extend the work from Assignment 1 and 2 to speed up the transformer model for more efficient training and inference. You will focus on optimizing the **Softmax** and **LayerNorm** batch reduction operations by writing custom CUDA code.

The CUDA optimizations are based on methods from the **LightSeq** and **LightSeq2** papers. We strongly encourage you to refer to the papers and the relevant lecture slides while working on this assignment. Before starting to write the CUDA code, make sure you have read through the write-up and understood what each kernel is doing.

---

## Setting up the Code

The starting codebase is provided in the following repository:  
[https://github.com/llmsystem/llmsys_f25_hw4](https://github.com/llmsystem/llmsys_f25_hw4)

You will need to merge it with your implementation from the previous assignments. Here are our suggested steps:

### Step 1: Install Requirements

Install the required dependencies and miniTorch by running the following commands:

```bash
pip install -r requirements.extra.txt
pip install -r requirements.txt
pip install -e .
```

### Step 2: Copy and Compile CUDA Kernels

Copy the CUDA kernel file `combine.cu` from Assignment 3 and compile it:

```bash
# From Assignment 2 to the current directory
cp <path_to_assignment_3>/combine.cu src/combine.cu

# Compile the CUDA kernels
bash compile_cuda.sh
```

### Step 3: Copy Files from Assignment 1

Copy `autodiff.py` from Assignment 1 to the specified location:

```bash
cp <path_to_assignment_1>/autodiff.py minitorch/autodiff.py
```

> **Hints for Copying and Pasting**:  
> When copying the `backpropagate()` function, make sure to double-check the implementation you wrote for Assignment 1.  
> It's a good idea to set a default value of `0.0` before accumulating the gradients in `backpropagate()`.  
> We recommend initializing the derivatives map for each `unique_id` to `0.0` outside the for loop.

### Step 4: Incrementally Add Functions

Keep copying several other functions from Assignment 3 as needed to complete this assignment.

- Copy `nn.py`:
  ```bash
  cp <path_to_assignment_3>/nn.py minitorch/nn.py
  ```

- Copy `modules_basic.py`:
  ```bash
  cp <path_to_assignment_3>/modules_basic.py minitorch/modules_basic.py
  ```

- Copy `modules_transfomer.py`:
  ```bash
  cp <path_to_assignment_3>/modules_transfomer.py minitorch/modules_transfomer.py
  ```

- Copy `run_machine_translation.py`:
  ```bash
  cp <path_to_assignment_3>/run_machine_translation.py run_machine_translation.py
  ```

---

## Problem 1: Softmax Optimization (40 points)

### Problem 1.1: Softmax Forward (20 points)

In this part, you will implement a **fused kernel** of softmax in the attention mechanism.

The softmax function for a vector $ \mathbf{x} $ is defined as:

$$
\text{softmax}(\mathbf{x})_i = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$

where $ x_i $ is the $ i $-th element of $ \mathbf{x} $.

The kernel also incorporates attention mechanisms, particularly in its handling of **attention masks**. Attention masks are used to control the model's focus on specific parts of the input.

#### Instructions

##### Implement the CUDA Kernel

Write the CUDA kernel for softmax in `src/softmax_kernel.cu`:

```cpp
template <typename T, int block_dim, int ele_per_thread>
__global__ void ker_attn_softmax(T *inp, ...) {
    ...
}

template <typename T, int block_dim, int ele_per_thread>
__global__ void ker_attn_softmax_lt32(T *inp, ...) {
    ...
}
```

> **Note**: The `ker_attn_softmax_lt32` kernel is already implemented for sequences shorter than 32 and does not require block-level parallelism.  
> Review the provided implementation of `ker_attn_softmax_lt32` and use it as a reference to implement `ker_attn_softmax`.

##### Compile the CUDA File

Compile the CUDA file using the following command:

```bash
nvcc -o minitorch/cuda_kernels/softmax_kernel.so --shared src/softmax_kernel.cu -Xcompiler -fPIC
```

##### Bind the Kernel to miniTorch

In `minitorch/cuda_kernel_ops.py`, bind the kernel by passing the CUDA stream:

```python
class CudaKernelOps(TensorOps):
    @staticmethod
    def attn_softmax_fw(inp: Tensor, mask: Tensor):
        ...
```

In `minitorch/tensor_functions.py`, define the softmax forward function:

```python
class Attn_Softmax(Function):
    @staticmethod
    def forward(ctx: Context, inp: Tensor, mask: Tensor) -> Tensor:
        ...
```

##### Test the Implementation

Run the provided test script and ensure your implementation achieves an average speedup of approximately **6.5×**:

```bash
python kernel_tests/test_softmax_fw.py
```

---

### Understanding Softmax Forward Kernels

The `ker_attn_softmax_lt32` and `ker_attn_softmax` kernels are optimized for different input sizes:

#### `ker_attn_softmax_lt32`:
- Utilizes **warp-level primitives** for reduction.
- Suitable for smaller input sizes (**<32 sequence length**).
- Efficient parallel reduction without block-wide synchronization.

#### `ker_attn_softmax`:
- Employs **block-level reduction techniques**.
- Suitable for larger input sizes.
- Includes **two phases of reduction** (max and sum) followed by a normalization step with synchronization.

---

### Algorithmic Steps

The softmax computation in both kernels consists of three main steps:

1. **Compute Max**
   - Identify the maximum value for normalization to avoid numerical overflow during exponentiation.

2. **Compute Exponentials and Sum**
   - Calculate the exponentials of normalized values and their sum for final normalization.

3. **Compute Final Result**
   - Normalize the exponentials by dividing by the sum to obtain softmax probabilities.
   - Use **CUB library's `BlockStore`** to minimize memory transactions.

---

### Computing the Maximum Value for Normalization

Both kernels implement the maximum value computation in the same way. Study the implementation in `ker_attn_softmax_lt32`.

#### Steps to Compute the Maximum Value:

1. **Thread Local Max**:
   - Each thread computes a local maximum.
   - Intermediate values and attention mask adjustments are stored in `val[token_per_reduce][ele_per_thread]`.
   - The thread-local maximum is recorded in `l_max[token_per_reduce]`, initialized to `REDUCE_FLOAT_INF_NEG`.

2. **Future Token Masking**:
   - If future tokens are masked, their values are excluded from the maximum computation by setting them to `REDUCE_FLOAT_INF_NEG`.

3. **Attention Mask Adjustment**:
   - Adjust the input value by adding the corresponding attention mask value.

4. **Iterative Updates**:
   - Update the thread-local maximum using `fmaxf`.

#### Block-Level Reduction for Global Max:
   - `ker_attn_softmax_lt32`: Uses a warp-level reduction with a custom `warpReduce` function.
   - `ker_attn_softmax`: Uses block-wide reduction with the CUB library's `BlockLoad` and shared memory for synchronization.

---

### Problem 1.2: Softmax Backward (20 points)

The gradient of the softmax function for a vector $ \mathbf{y} = \text{softmax}(\mathbf{x}) $ is given by:

$$
\frac{\partial y_i}{\partial x_j} = y_i (\delta_{ij} - y_j)
$$

where $ \delta_{ij} $ is the Kronecker delta.

#### Implement `launch_attn_softmax_bw` in `src/softmax_kernel.cu`:

```cpp
void launch_attn_softmax_bw(float *out_grad,
                            const float *soft_inp, 
                            int rows,
                            int softmax_len,
                            cudaStream_t stream)
```

> In lectures, we described the use of templates for tuning kernel parameters.  
> When implementing `launch_attn_softmax_bw`, you should compute the `ITERATIONS` parameter of `ker_attn_softmax_bw` depending on different max sequence lengths in:
>
> `{32, 64, 128, 256, 384, 512, 768, 1024, 2048}`

**Hint**: Refer to the way templates are used in `launch_attn_softmax`.

```cpp
template <typename T, int ITERATIONS>
__global__ void ker_attn_softmax_bw(T *grad, ...) {
    ...
}
```

#### Compile the CUDA file:

```bash
nvcc -o minitorch/cuda_kernels/softmax_kernel.so --shared src/softmax_kernel.cu -Xcompiler -fPIC
```

#### Bind the kernel with miniTorch in `minitorch/cuda_kernel_ops.py`:

**Hint**: You should pass the CUDA stream to the function, define it with:

```python
stream_1 = torch.cuda.current_stream().cuda_stream
```

```python
class CudaKernelOps(TensorOps):
    @staticmethod
    def attn_softmax_bw(out_grad: Tensor, soft_inp: Tensor):
        ...
```

And in `minitorch/tensor_functions.py`:

```python
class Attn_Softmax(Function):
    @staticmethod
    def backward(ctx: Context, out_grad: Tensor) -> Tensor:
        ...
```

#### Pass the test and notice an average speedup around **0.5×** with our given default max lengths `{32, 64, 128, 256, 384, 512, 768, 1024, 2048}`.  
You can try other setups of max length and achieve a higher speedup, but it will not be graded.

```bash
python kernel_tests/test_softmax_bw.py
```

---

### Understanding Softmax Backward Kernel

The `ker_attn_softmax_bw` function is a CUDA kernel for computing the backward pass of the softmax function in self-attention mechanisms. Here are the steps:

#### 1. Initialization
- The function calculates the backward pass for each element in the gradient and the output of the softmax forward pass.
- The grid and block dimensions are configured based on the batch size, number of heads, and sequence length.

#### 2. Gradient Calculation
- The function iterates over the softmax length, with each thread handling a portion of the data.
- It loads the gradient and input (output of softmax forward) into registers.
- A **local sum** is computed for each thread, which is a key part of the gradient calculation for softmax.

#### 3. Gradient Computation
- The sum is shared across the warp using **warp shuffle operations**.
- The final gradient for each element is computed by modifying the forward pass output with the computed sum.

---

# Problem 2: LayerNorm Optimization (40 points)

## Problem 2.1: LayerNorm Forward (20 points)

Layer Normalization (LayerNorm) normalizes the input $ \mathbf{x} $ by:

$$
\text{LayerNorm}(\mathbf{x}) = \gamma \cdot \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

where:
- $ \mu $ and $ \sigma $ are the **mean** and **standard deviation** of $ \mathbf{x} $,
- $ \gamma $ and $ \beta $ are learnable affine transform parameters.

To optimize performance, we avoid sequential reductions. Instead of computing variance as $ \sigma^2 = \mathbb{E}[x^2] - \mu^2 $, we compute it using:

$$
\sigma^2 = \frac{1}{N} \sum x_i^2 - \mu^2
$$

This allows **concurrent computation** of the means of $ \mathbf{x} $ and $ \mathbf{x}^2 $, enabling better parallelization.

### Steps

#### Implement the CUDA Kernel

In `src/layernorm_kernel.cu`, implement the forward kernel:

```cpp
template <typename T>
__global__ void ker_layer_norm(T *ln_res, const T *inp, const T *gamma, const T *beta,
                               int rows, int cols, float epsilon) {
    ...
}
```

Key considerations:
- Use **float4** vectorization to process 4 floats per thread per load for higher throughput.
- Each thread computes partial sums of $ x $ and $ x^2 $ over its assigned elements.
- Perform **block-level reduction** using shared memory and `__syncthreads()` or CUB’s `BlockReduce`.
- Compute final mean and variance, then apply normalization and affine transformation.

#### Compile the CUDA File

```bash
nvcc -o minitorch/cuda_kernels/layernorm_kernel.so --shared src/layernorm_kernel.cu -Xcompiler -fPIC
```

#### Bind the Kernel with miniTorch

In `minitorch/cuda_kernel_ops.py`:

```python
stream_1 = torch.cuda.current_stream().cuda_stream

class CudaKernelOps(TensorOps):
    @staticmethod
    def layernorm_fw(inp: Tensor, gamma: Tensor, beta: Tensor) -> Tensor:
        # Launch ker_layer_norm with CUDA stream
        ...
```

In `minitorch/tensor_functions.py`:

```python
class LayerNorm(Function):
    @staticmethod
    def forward(ctx: Context, inp: Tensor, gamma: Tensor, beta: Tensor, eps: float = 1e-5) -> Tensor:
        ctx.save_for_backward(inp, gamma, beta)
        ctx.eps = eps
        return CudaKernelOps.layernorm_fw(inp, gamma, beta)
```

#### Test the Implementation

Run the test script:

```bash
python kernel_tests/test_layernorm_fw.py
```

Run the provided test script and ensure your implementation achieves an average speedup of approximately **15.8×**:.

---

### Understanding LayerNorm Forward Kernels

The key optimization lies in **SIMD-style processing via `float4`** and efficient **block-wide reductions**.

#### Use of `float4` for Speedup

- `reinterpret_cast<float4*>` is used to cast the input array to vectorized form:
  ```cpp
  float4 *inp_f4 = reinterpret_cast<float4*>(inp);
  ```
- Each thread processes multiple elements at once, reducing memory transactions and increasing arithmetic intensity.

#### Algorithmic Steps

1. **Compute Sums of $ \mathbf{x} $ and $ \mathbf{x}^2 $**:
   - Load data using `float4` for coalesced memory access.
   - Accumulate local sums of $ x $ and $ x^2 $ in registers.

2. **Reduction Across Threads**:
   - Use shared memory and `blockReduce` (or CUB's `BlockReduce`) to compute global sum.
   - Add small epsilon (`LN_EPSILON`) to variance for numerical stability.

3. **Final Normalization**:
   - Compute $ \mu = \text{sum\_x} / N $
   - Compute $ \sigma^2 = \text{sum\_x2} / N - \mu^2 $
   - Normalize: $ \hat{x} = (x - \mu) / \sqrt{\sigma^2 + \epsilon} $
   - Apply affine transform: $ y = \gamma \cdot \hat{x} + \beta $

---

## Problem 2.2: LayerNorm Backward (20 points)

Let:
- $ z = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} $
- $ dy $: output gradient
- $ dx $: input gradient

Then the gradient with respect to input $ x $ is:

$$
\frac{\partial \mathcal{L}}{\partial x_i} = \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}} \left( dy_i - \frac{1}{N} \sum_{j=1}^{N} \left[ dy_j + (x_j - \mu) \cdot \frac{dy_j}{\sigma^2 + \epsilon} \right] \right)
$$

Alternatively, this can be decomposed into two batch reductions:

$$
dx = \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}} \left( dy - \frac{1}{N} (\text{reduce\_sum}(dy) + z \cdot \text{reduce\_sum}(dy \cdot z)) \right)
$$

Speedup is achieved by **concurrently computing**:
- $ \text{reduce\_sum}(dy) $
- $ \text{reduce\_sum}(dy \cdot z) $

Gradients for learnable parameters:
$$
\frac{\partial \mathcal{L}}{\partial \gamma} = dy \cdot z, \quad
\frac{\partial \mathcal{L}}{\partial \beta} = dy
$$

### Steps to Implement

#### Implement CUDA Kernels in `src/layernorm_kernel.cu`

```cpp
// Compute input gradient
template <typename T>
__global__ void ker_ln_bw_dinp(T *inp_grad, const T *out_grad, const T *inp,
                               const T *gamma, const T *means, const T *vars,
                               int rows, int cols, float epsilon) {
    ...
}

// Compute gamma and beta gradients
template <typename T>
__global__ void ker_ln_bw_dgamma_dbetta(T *gamma_grad, T *beta_grad,
                                        const T *out_grad, const T *inp,
                                        const T *means, const T *vars,
                                        int rows, int cols, float epsilon) {
    ...
}
```

#### Compile the CUDA File

```bash
nvcc -o minitorch/cuda_kernels/layernorm_kernel.so --shared src/layernorm_kernel.cu -Xcompiler -fPIC
```

#### Bind the Kernel with miniTorch

In `minitorch/cuda_kernel_ops.py`:

```python
stream_1 = torch.cuda.current_stream().cuda_stream

class CudaKernelOps(TensorOps):
    @staticmethod
    def layernorm_bw(out_grad: Tensor, inp: Tensor, gamma: Tensor, beta: Tensor, eps: float):
        # Launch both backward kernels using CUDA stream
        ...
```

In `minitorch/tensor_functions.py`:

```python
class LayerNorm(Function):
    @staticmethod
    def backward(ctx: Context, out_grad: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        inp, gamma, beta = ctx.saved_values
        eps = ctx.eps
        return CudaKernelOps.layernorm_bw(out_grad, inp, gamma, beta, eps)
```

#### Test the Implementation

```bash
python kernel_tests/test_layernorm_bw.py
```

You should observe an **average speedup of approximately 3.7×**.

---

### Understanding LayerNorm Backward Kernels

#### Input Gradient (`ker_ln_bw_dinp`)

##### Initialization
- Each thread handles a subset of elements in the input tensor.

##### Algorithmic Steps
1. Use `reinterpret_cast<float4*>` to load `out_grad` and `inp` efficiently.
2. Compute normalized value $ z = (x - \mu) / \sqrt{\sigma^2 + \epsilon} $.
3. Compute local contributions to:
   - $ \text{sum}_1 = \sum dy $
   - $ \text{sum}_2 = \sum dy \cdot z $
4. Perform block-level reduction using `blockReduce` or `shfl_down`.
5. Reconstruct final input gradient using the formula above.

#### Gamma and Beta Gradients (`ker_ln_bw_dgamma_dbetta`)

##### Initialization
- Declare shared memory arrays:
  ```cpp
  __shared__ float betta_buffer[32][32];
  __shared__ float gamma_buffer[32][32];
  ```
- Use Cooperative Groups for fine-grained synchronization.

##### Loop Over Rows
- Threads in the y-dimension loop over rows (for large tensors).
- Compute partial gradients:  
  $ d\beta_i += dy_i $,  
  $ d\gamma_i += dy_i \cdot z_i $

##### Shared Memory Storage
- Store partial results in tiled layout in shared memory to avoid bank conflicts.

##### Reduction within Thread Block
- Use **warp-level shuffle operations**:
  ```cpp
  g.shfl_down()
  ```
- Eliminates need for shared memory in reduction phase.
- Enables **warp-level reduction without bank conflicts**.

##### Final Result Assignment
- Thread with `thread_rank() == 0` writes reduced result to global memory.

---

### Key Insight: `g.shfl_down` and Cooperative Groups

As noted in the [NVIDIA blog](https://developer.nvidia.com/blog/cooperative-groups/):

> "Using `thread_block_tile::shfl_down()` to simplify our warp-level reduction does benefit our code: it simplifies it and eliminates the need for shared memory."

#### What `g.shfl_down` Does
- Allows a thread to **read a value from another thread within the same warp**.
- Specifically, `shfl_down(val, offset)` gets the value of `val` from the thread `offset` positions below.
- Used in reductions: each step, higher threads add their value to lower ones.

#### Why It’s Better
- **No shared memory required** → saves memory bandwidth and avoids bank conflicts.
- **Faster synchronization** → no `__syncthreads()` needed.
- **Compiler optimizations** possible when tile size is known at compile time (e.g., `thread_block_tile<32>`).

#### Example Usage

```cpp
auto g = cg::tiled_partition<32>(cg::this_thread_block());
float val = /* some value */;
for (int i = 16; i >= 1; i /= 2) {
    val += g.shfl_down(val, i);
}
// Now val in thread 0 contains the sum of all 32 values in the warp
```

This replaces a full shared-memory-based reduction with a **lean, fast, memory-efficient** version.

---

# Problem 3: Adopt Fused Kernels in Transformer (20 points)

Now that the optimized CUDA kernels for **Softmax** and **LayerNorm** are implemented and bound to miniTorch, it's time to integrate them into the transformer model.

### Task

Replace the default **Softmax** and **LayerNorm** operations in the following components in `minitorch/modules_transfomer.py`:
- `MultiHeadAttention`
- `TransformerLayer`
- `DecoderLM`

Use the **accelerated kernels** via the `CudaKernelOps` interface when `--use-fused-kernel True`.

Example:
```python
# Instead of:
out = softmax(x, dim=-1)

# Use:
if use_fused_kernel:
    out = CudaKernelOps.attn_softmax_fw(x, mask)
else:
    out = softmax(x, dim=-1)
```

Similarly for LayerNorm:
```python
# Use:
if use_fused_kernel:
    out = CudaKernelOps.layernorm_fw(x, gamma, beta)
else:
    out = layer_norm(x, gamma, beta)
```

Ensure the flag `--use-fused-kernel` is properly passed through the model.

### Benchmark Performance

Train the transformer for **one epoch** with and without fused kernels:

```bash
python project/run_machine_translation.py --use-fused-kernel False
python project/run_machine_translation.py --use-fused-kernel True
```

Record the running times.

### Expected Result

According to **Amdahl's Law**, since only **Softmax** and **LayerNorm** are optimized (not the entire model), the overall speedup will be limited by the fraction of time spent in these operations.

However, you should still observe an **average speedup of approximately 1.1×**.

---

# Submission

Please submit the entire `llmsys_s25_hw4` folder as a **zip file** on Canvas.

---
