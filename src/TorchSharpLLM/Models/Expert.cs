using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace TorchSharpLLM.Models;

/// <summary>
/// A single expert with variable-depth feed-forward sub-layers.
/// Each sub-layer is: RMSNorm → Linear(hidden→ff) → SiLU gate → Linear(ff→hidden)
/// Supports optional LoRA adapters on any sub-layer.
/// </summary>
public sealed class Expert : Module<Tensor, Tensor>
{
    private readonly ModuleList<ExpertSubLayer> _subLayers;
    private readonly int _hiddenDim;
    private readonly int _ffDim;

    public int ExpertId { get; }
    public int Depth => (int)_subLayers.Count;

    public Expert(int expertId, int hiddenDim, int ffDim, int depth, double dropoutRate)
        : base($"Expert_{expertId}")
    {
        ExpertId = expertId;
        _hiddenDim = hiddenDim;
        _ffDim = ffDim;

        var layers = new List<ExpertSubLayer>();
        for (int i = 0; i < depth; i++)
            layers.Add(new ExpertSubLayer(hiddenDim, ffDim, dropoutRate, $"sub_{i}"));

        _subLayers = new ModuleList<ExpertSubLayer>(layers.ToArray());
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var x = input;
        foreach (var layer in _subLayers)
        {
            var prev = x;
            x = layer.forward(x);
            if (!ReferenceEquals(prev, input))
                prev.Dispose();
        }
        return x;
    }

    /// <summary>
    /// Add additional sub-layers to this expert (grow its depth).
    /// New layers are initialized to approximate an identity function.
    /// </summary>
    public void AddSubLayers(int count, double dropoutRate)
    {
        int currentDepth = Depth;
        for (int i = 0; i < count; i++)
        {
            var layer = new ExpertSubLayer(_hiddenDim, _ffDim, dropoutRate, $"sub_{currentDepth + i}");
            layer.InitAsIdentity();
            _subLayers.append(layer);
        }
    }

    /// <summary>
    /// Apply LoRA adapters to all sub-layers of this expert.
    /// </summary>
    public void ApplyLoRA(int rank = 16, double alpha = 32.0, double loraDropout = 0.05)
    {
        foreach (var layer in _subLayers)
            layer.ApplyLoRA(rank, alpha, loraDropout);
    }

    /// <summary>
    /// Merge all LoRA weights into base weights (for inference).
    /// </summary>
    public void MergeLoRA()
    {
        foreach (var layer in _subLayers)
            layer.MergeLoRA();
    }

    /// <summary>
    /// Un-merge LoRA weights (to resume training).
    /// </summary>
    public void UnmergeLoRA()
    {
        foreach (var layer in _subLayers)
            layer.UnmergeLoRA();
    }

    /// <summary>
    /// Freeze/unfreeze all parameters of this expert.
    /// </summary>
    public void SetTrainable(bool trainable)
    {
        foreach (var p in parameters())
            p.requires_grad_(trainable);
    }

    /// <summary>
    /// Freeze base weights but keep LoRA parameters trainable.
    /// </summary>
    public void FreezeBaseKeepLoRA()
    {
        foreach (var layer in _subLayers)
            layer.FreezeBaseKeepLoRA();
    }
}

/// <summary>
/// Single feed-forward sub-layer within an expert:
/// RMSNorm → GatedFFN (W_gate, W_up with SiLU gate, W_down) with optional LoRA.
/// </summary>
public sealed class ExpertSubLayer : Module<Tensor, Tensor>
{
    private readonly RMSNorm _norm;
    private Linear _wGate;    // hidden → ff (gate pathway)
    private Linear _wUp;      // hidden → ff (value pathway)
    private Linear _wDown;    // ff → hidden

    private LoRALinear? _loraGate;
    private LoRALinear? _loraUp;
    private LoRALinear? _loraDown;

    private readonly Dropout _dropout;
    private readonly int _hiddenDim;
    private readonly int _ffDim;
    private bool _hasLoRA;

    public ExpertSubLayer(int hiddenDim, int ffDim, double dropoutRate, string name)
        : base(name)
    {
        _hiddenDim = hiddenDim;
        _ffDim = ffDim;
        _hasLoRA = false;

        _norm = new RMSNorm(hiddenDim);
        _wGate = Linear(hiddenDim, ffDim, hasBias: false);
        _wUp = Linear(hiddenDim, ffDim, hasBias: false);
        _wDown = Linear(ffDim, hiddenDim, hasBias: false);
        _dropout = Dropout(dropoutRate);

        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        using var normed = _norm.forward(input);

        Tensor gate, up;
        if (_hasLoRA)
        {
            gate = _loraGate!.forward(normed);
            up = _loraUp!.forward(normed);
        }
        else
        {
            gate = _wGate.forward(normed);
            up = _wUp.forward(normed);
        }

        using var gateActivated = torch.nn.functional.silu(gate);
        using var gated = gateActivated * up;
        up.Dispose();

        Tensor down;
        if (_hasLoRA)
            down = _loraDown!.forward(gated);
        else
            down = _wDown.forward(gated);

        using var dropped = _dropout.forward(down);
        return input + dropped;
    }

    /// <summary>
    /// Initialize weights so this sub-layer approximates identity (for newly added layers).
    /// </summary>
    public void InitAsIdentity()
    {
        using (torch.no_grad())
        {
            // Zero out the down-projection so residual pass-through dominates
            nn.init.zeros_(_wDown.weight!);
        }
    }

    public void ApplyLoRA(int rank, double alpha, double loraDropout)
    {
        if (_hasLoRA) return;

        _loraGate = LoRALinear.FromExisting(_wGate, rank, alpha, loraDropout);
        _loraUp = LoRALinear.FromExisting(_wUp, rank, alpha, loraDropout);
        _loraDown = LoRALinear.FromExisting(_wDown, rank, alpha, loraDropout);

        register_module("loraGate", _loraGate);
        register_module("loraUp", _loraUp);
        register_module("loraDown", _loraDown);

        _hasLoRA = true;
    }

    public void MergeLoRA()
    {
        if (!_hasLoRA) return;
        _loraGate!.MergeWeights();
        _loraUp!.MergeWeights();
        _loraDown!.MergeWeights();
    }

    public void UnmergeLoRA()
    {
        if (!_hasLoRA) return;
        _loraGate!.UnmergeWeights();
        _loraUp!.UnmergeWeights();
        _loraDown!.UnmergeWeights();
    }

    public void FreezeBaseKeepLoRA()
    {
        // Freeze base linear weights
        _wGate.weight!.requires_grad_(false);
        _wUp.weight!.requires_grad_(false);
        _wDown.weight!.requires_grad_(false);

        // Keep LoRA trainable
        if (_hasLoRA)
        {
            _loraGate!.SetLoraTrainable(true);
            _loraUp!.SetLoraTrainable(true);
            _loraDown!.SetLoraTrainable(true);
        }
    }
}
