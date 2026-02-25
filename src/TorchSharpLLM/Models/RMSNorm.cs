using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace TorchSharpLLM.Models;

/// <summary>
/// Root Mean Square Layer Normalization — lighter and more stable than LayerNorm.
/// </summary>
public sealed class RMSNorm : Module<Tensor, Tensor>
{
    private readonly Parameter _weight;
    private readonly double _eps;

    public RMSNorm(int dim, double eps = 1e-6) : base("RMSNorm")
    {
        _eps = eps;
        _weight = Parameter(torch.ones(dim));
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        using var variance = input.to_type(ScalarType.Float32).pow(2).mean(-1, keepdim: true);
        using var normed = input * (variance + _eps).rsqrt();
        return _weight * normed.to_type(input.dtype);
    }
}
