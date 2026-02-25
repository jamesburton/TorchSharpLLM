using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace TorchSharpLLM.Models;

/// <summary>
/// A single MoE layer: routes each token through top-k experts,
/// combines their outputs weighted by the gate, and adds the residual.
/// </summary>
public sealed class MoELayer : Module<Tensor, (Tensor output, Tensor auxLoss)>
{
    private readonly MoERouter _router;
    private readonly ModuleList<Expert> _experts;
    private readonly RMSNorm _norm;

    public MoERouter Router => _router;
    public ModuleList<Expert> Experts => _experts;

    public MoELayer(ModelConfig config, int layerIndex) : base($"MoELayer_{layerIndex}")
    {
        _router = new MoERouter(
            config.HiddenDim,
            config.NumExperts,
            config.TopK,
            config.LoadBalanceLossWeight,
            config.RouterZLossWeight);

        var experts = new List<Expert>();
        for (int i = 0; i < config.NumExperts; i++)
        {
            int depth = config.GetExpertDepth(i);
            experts.Add(new Expert(i, config.HiddenDim, config.FeedForwardDim, depth, config.DropoutRate));
        }
        _experts = new ModuleList<Expert>(experts.ToArray());

        _norm = new RMSNorm(config.HiddenDim);
        RegisterComponents();
    }

    public override (Tensor output, Tensor auxLoss) forward(Tensor input)
    {
        var residual = input;
        using var normed = _norm.forward(input);

        // Route tokens
        var routing = _router.forward(normed);
        long numTokens = routing.BatchSize * routing.SeqLen;
        int hiddenDim = (int)normed.shape[^1];
        int topK = (int)routing.GateWeights.shape[^1];

        // Flatten input for per-token processing
        using var flatInput = normed.view(numTokens, hiddenDim);

        // Accumulate weighted expert outputs
        var combinedOutput = torch.zeros(numTokens, hiddenDim,
            dtype: flatInput.dtype, device: flatInput.device);

        // Process each expert: gather tokens routed to it, run expert, scatter back
        int numExperts = (int)_experts.Count;
        for (int k = 0; k < topK; k++)
        {
            using var kIndices = routing.ExpertIndices[.., k]; // [numTokens]
            using var kWeights = routing.GateWeights[.., k];   // [numTokens]

            for (int e = 0; e < numExperts; e++)
            {
                // Find tokens assigned to expert e at position k
                using var mask = kIndices.eq(e);
                using var tokenPositions = mask.nonzero().squeeze(-1);

                if (tokenPositions.numel() == 0) continue;

                // Gather tokens for this expert
                using var expertInput = flatInput.index_select(0, tokenPositions);

                // Run expert
                using var expertOutput = _experts[e].forward(expertInput);

                // Weight by gate probability
                using var expertWeights = kWeights.index_select(0, tokenPositions).unsqueeze(-1);
                using var weightedOutput = expertOutput * expertWeights;

                // Scatter-add back
                combinedOutput.index_add_(0, tokenPositions, weightedOutput);
            }
        }

        // Reshape back and add residual
        using var reshaped = combinedOutput.view(routing.BatchSize, routing.SeqLen, hiddenDim);

        routing.GateWeights.Dispose();
        routing.ExpertIndices.Dispose();
        routing.RouterLogits.Dispose();

        return (residual + reshaped, routing.AuxLoss);
    }
}
