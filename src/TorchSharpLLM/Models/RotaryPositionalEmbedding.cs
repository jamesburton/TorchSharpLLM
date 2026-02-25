using TorchSharp;
using static TorchSharp.torch;

namespace TorchSharpLLM.Models;

/// <summary>
/// Rotary Positional Embedding (RoPE) — applies rotation to Q/K in attention.
/// </summary>
public static class RotaryPositionalEmbedding
{
    /// <summary>
    /// Pre-compute the sin/cos rotation tables for a given sequence length and head dimension.
    /// </summary>
    public static (Tensor cos, Tensor sin) Precompute(int seqLen, int headDim, Device? device = null, double theta = 10000.0)
    {
        device ??= torch.CPU;

        // freq_i = 1 / theta^(2i / headDim)  for i in [0, headDim/2)
        using var exponents = torch.arange(0, headDim, 2, dtype: ScalarType.Float32, device: device) / headDim;
        using var freqs = torch.pow(theta, -exponents); // [headDim/2]

        using var positions = torch.arange(seqLen, dtype: ScalarType.Float32, device: device); // [seqLen]
        using var angles = torch.outer(positions, freqs); // [seqLen, headDim/2]

        var cos = angles.cos(); // [seqLen, headDim/2]
        var sin = angles.sin(); // [seqLen, headDim/2]

        return (cos, sin);
    }

    /// <summary>
    /// Apply RoPE rotation to a Q or K tensor.
    /// x: [batch, heads, seqLen, headDim]
    /// </summary>
    public static Tensor Apply(Tensor x, Tensor cos, Tensor sin)
    {
        int halfDim = (int)x.shape[^1] / 2;

        // Split last dim into two halves
        using var x1 = x[.., .., .., ..halfDim];
        using var x2 = x[.., .., .., halfDim..];

        // Reshape cos/sin to broadcast: [1, 1, seqLen, halfDim]
        int seqLen = (int)x.shape[^2];
        using var cosSlice = cos[..seqLen].unsqueeze(0).unsqueeze(0);
        using var sinSlice = sin[..seqLen].unsqueeze(0).unsqueeze(0);

        // Rotate
        using var rotated1 = x1 * cosSlice - x2 * sinSlice;
        using var rotated2 = x1 * sinSlice + x2 * cosSlice;

        return torch.cat(new[] { rotated1, rotated2 }, dim: -1);
    }
}
