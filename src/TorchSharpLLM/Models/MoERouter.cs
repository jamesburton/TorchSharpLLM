using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace TorchSharpLLM.Models;

/// <summary>
/// Mixture-of-Experts router with top-k gating, load-balancing auxiliary loss,
/// and router z-loss for training stability.
/// </summary>
public sealed class MoERouter : Module<Tensor, MoERouter.RoutingResult>
{
    private Linear _gate;
    private readonly int _topK;
    private readonly double _loadBalanceWeight;
    private readonly double _zLossWeight;
    private int _numExperts;

    public int NumExperts => _numExperts;

    public MoERouter(int hiddenDim, int numExperts, int topK, double loadBalanceWeight, double zLossWeight)
        : base("MoERouter")
    {
        _numExperts = numExperts;
        _topK = topK;
        _loadBalanceWeight = loadBalanceWeight;
        _zLossWeight = zLossWeight;

        _gate = Linear(hiddenDim, numExperts, hasBias: false);
        RegisterComponents();
    }

    public override RoutingResult forward(Tensor input)
    {
        // input: [batch, seqLen, hiddenDim]
        long batch = input.shape[0];
        long seqLen = input.shape[1];

        // Flatten to [batch * seqLen, hiddenDim]
        using var flat = input.view(batch * seqLen, -1);

        // Router logits: [tokens, numExperts]
        var logits = _gate.forward(flat);

        // Top-k selection
        var (topkWeights, topkIndices) = logits.topk(_topK, dim: -1);
        var gateWeights = torch.nn.functional.softmax(topkWeights, dim: -1);

        // Compute auxiliary losses for training
        var auxLoss = ComputeAuxLoss(logits, topkIndices, batch * seqLen);

        return new RoutingResult
        {
            GateWeights = gateWeights,         // [tokens, topK]
            ExpertIndices = topkIndices,        // [tokens, topK]
            AuxLoss = auxLoss,
            RouterLogits = logits,
            BatchSize = batch,
            SeqLen = seqLen
        };
    }

    private Tensor ComputeAuxLoss(Tensor logits, Tensor topkIndices, long numTokens)
    {
        // Load balancing loss: encourages uniform expert usage
        // fraction of tokens routed to each expert
        using var oneHot = torch.nn.functional.one_hot(topkIndices, _numExperts).to_type(ScalarType.Float32);
        using var tokenFraction = oneHot.sum(new long[] { 0, 1 }) / (numTokens * _topK); // [numExperts]

        // average routing probability per expert
        using var routerProbs = torch.nn.functional.softmax(logits, dim: -1);
        using var avgProbs = routerProbs.mean(dim: 0); // [numExperts]

        using var balanceLoss = (tokenFraction * avgProbs).sum() * _numExperts * _loadBalanceWeight;

        // Router z-loss: penalizes large logits for training stability
        using var zLoss = logits.pow(2).mean() * _zLossWeight;

        return balanceLoss + zLoss;
    }

    /// <summary>
    /// Resize the router to accommodate a new number of experts.
    /// Preserves existing expert gate weights where possible.
    /// </summary>
    public void ResizeExperts(int newNumExperts, int hiddenDim)
    {
        if (newNumExperts == _numExperts) return;

        var oldGate = _gate;
        var newGate = Linear(hiddenDim, newNumExperts, hasBias: false);

        using (torch.no_grad())
        {
            int copyCount = Math.Min(_numExperts, newNumExperts);
            newGate.weight!.index_copy_(0,
                torch.arange(copyCount, device: newGate.weight.device),
                oldGate.weight![..copyCount]);
        }

        _gate = newGate;
        register_module("_gate", _gate);
        _numExperts = newNumExperts;

        oldGate.Dispose();
    }

    public sealed class RoutingResult
    {
        public required Tensor GateWeights { get; init; }
        public required Tensor ExpertIndices { get; init; }
        public required Tensor AuxLoss { get; init; }
        public required Tensor RouterLogits { get; init; }
        public required long BatchSize { get; init; }
        public required long SeqLen { get; init; }
    }
}
