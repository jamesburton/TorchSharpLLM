using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace TorchSharpLLM.Models;

/// <summary>
/// Multi-head self-attention with RoPE and causal masking.
/// </summary>
public sealed class MultiHeadAttention : Module<Tensor, Tensor, Tensor, Tensor>
{
    private readonly Linear _qProj;
    private readonly Linear _kProj;
    private readonly Linear _vProj;
    private readonly Linear _outProj;
    private readonly RMSNorm _norm;
    private readonly Dropout _dropout;

    private readonly int _numHeads;
    private readonly int _headDim;

    public MultiHeadAttention(int hiddenDim, int numHeads, double dropoutRate) : base("MultiHeadAttention")
    {
        _numHeads = numHeads;
        _headDim = hiddenDim / numHeads;

        _qProj = Linear(hiddenDim, hiddenDim, hasBias: false);
        _kProj = Linear(hiddenDim, hiddenDim, hasBias: false);
        _vProj = Linear(hiddenDim, hiddenDim, hasBias: false);
        _outProj = Linear(hiddenDim, hiddenDim, hasBias: false);
        _norm = new RMSNorm(hiddenDim);
        _dropout = Dropout(dropoutRate);

        RegisterComponents();
    }

    /// <param name="x">Input tensor [batch, seqLen, hiddenDim]</param>
    /// <param name="rosCos">RoPE cos table [maxSeq, headDim/2]</param>
    /// <param name="rosSin">RoPE sin table [maxSeq, headDim/2]</param>
    /// <returns>Output tensor [batch, seqLen, hiddenDim]</returns>
    public override Tensor forward(Tensor x, Tensor rosCos, Tensor rosSin)
    {
        var residual = x;
        x = _norm.forward(x);

        long batch = x.shape[0];
        long seqLen = x.shape[1];

        // Project Q, K, V and reshape to [batch, heads, seqLen, headDim]
        using var q = _qProj.forward(x).view(batch, seqLen, _numHeads, _headDim).transpose(1, 2);
        using var k = _kProj.forward(x).view(batch, seqLen, _numHeads, _headDim).transpose(1, 2);
        var v = _vProj.forward(x).view(batch, seqLen, _numHeads, _headDim).transpose(1, 2);

        // Apply RoPE to Q and K
        using var qRot = RotaryPositionalEmbedding.Apply(q, rosCos, rosSin);
        using var kRot = RotaryPositionalEmbedding.Apply(k, rosCos, rosSin);

        // Scaled dot-product attention with causal mask
        double scale = 1.0 / Math.Sqrt(_headDim);
        using var scores = torch.matmul(qRot, kRot.transpose(-2, -1)) * scale;

        // Causal mask: upper-triangular = -inf
        using var mask = torch.ones(seqLen, seqLen, dtype: scores.dtype, device: scores.device)
            .triu(diagonal: 1) * float.NegativeInfinity;
        using var maskedScores = scores + mask.unsqueeze(0).unsqueeze(0);
        using var attnWeights = torch.nn.functional.softmax(maskedScores, dim: -1);
        using var droppedWeights = _dropout.forward(attnWeights);

        // Weighted sum and reshape back
        using var attnOut = torch.matmul(droppedWeights, v)
            .transpose(1, 2)
            .contiguous()
            .view(batch, seqLen, _numHeads * _headDim);

        using var projected = _outProj.forward(attnOut);
        return residual + projected;
    }
}
