using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace TorchSharpLLM.Models;

/// <summary>
/// LoRA (Low-Rank Adaptation) wrapper around a frozen linear layer.
/// output = frozen_W(x) + (B @ A)(x) * (alpha / rank)
///
/// The original weight is frozen by default. Only A and B are trainable.
/// Call MergeWeights() to fuse LoRA into the base weight for inference speed,
/// and UnmergeWeights() to separate them again for further training.
/// </summary>
public sealed class LoRALinear : Module<Tensor, Tensor>
{
    private readonly Linear _base;
    private readonly Parameter _loraA;
    private readonly Parameter _loraB;
    private readonly Dropout _dropout;
    private readonly double _scaling;
    private bool _merged;

    public int Rank { get; }
    public int InFeatures { get; }
    public int OutFeatures { get; }

    public LoRALinear(int inFeatures, int outFeatures, int rank = 16, double alpha = 32.0, double dropout = 0.05)
        : base("LoRALinear")
    {
        Rank = rank;
        InFeatures = inFeatures;
        OutFeatures = outFeatures;
        _scaling = alpha / rank;
        _merged = false;

        _base = Linear(inFeatures, outFeatures, hasBias: false);
        // Freeze base weight
        _base.weight!.requires_grad_(false);

        // LoRA low-rank matrices: A projects down, B projects up
        _loraA = Parameter(torch.empty(rank, inFeatures));
        _loraB = Parameter(torch.zeros(outFeatures, rank));
        _dropout = Dropout(dropout);

        // Kaiming init for A
        nn.init.kaiming_uniform_(_loraA, a: Math.Sqrt(5));

        RegisterComponents();
    }

    /// <summary>
    /// Wrap an existing Linear layer with LoRA, freezing its weights.
    /// </summary>
    public static LoRALinear FromExisting(Linear existing, int rank = 16, double alpha = 32.0, double dropout = 0.05)
    {
        int inF = (int)existing.weight!.shape[1];
        int outF = (int)existing.weight.shape[0];

        var lora = new LoRALinear(inF, outF, rank, alpha, dropout);
        lora._base.weight!.copy_(existing.weight);
        lora._base.weight.requires_grad_(false);

        if (existing.bias is not null && lora._base.bias is not null)
            lora._base.bias.copy_(existing.bias);

        return lora;
    }

    public override Tensor forward(Tensor input)
    {
        if (_merged)
            return _base.forward(input);

        using var baseOut = _base.forward(input);
        using var dropped = _dropout.forward(input);
        // dropped @ A^T @ B^T  = dropped @ (B @ A)^T
        using var loraOut = torch.matmul(torch.matmul(dropped, _loraA.t()), _loraB.t()) * _scaling;
        return baseOut + loraOut;
    }

    /// <summary>
    /// Merge LoRA weights into the base linear layer for faster inference.
    /// </summary>
    public void MergeWeights()
    {
        if (_merged) return;
        using (torch.no_grad())
        {
            // W' = W + (B @ A) * scaling
            using var delta = torch.matmul(_loraB, _loraA) * _scaling;
            _base.weight!.add_(delta);
        }
        _merged = true;
    }

    /// <summary>
    /// Separate LoRA weights from the base layer for continued training.
    /// </summary>
    public void UnmergeWeights()
    {
        if (!_merged) return;
        using (torch.no_grad())
        {
            using var delta = torch.matmul(_loraB, _loraA) * _scaling;
            _base.weight!.sub_(delta);
        }
        _merged = false;
    }

    /// <summary>
    /// Freeze/unfreeze only the LoRA parameters (A and B).
    /// </summary>
    public void SetLoraTrainable(bool trainable)
    {
        _loraA.requires_grad_(trainable);
        _loraB.requires_grad_(trainable);
    }
}
