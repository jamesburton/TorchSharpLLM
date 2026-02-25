namespace TorchSharpLLM.Models;

/// <summary>
/// Configuration for the MoE Transformer model.
/// Default values target ~1.5GB total model size with 8 heterogeneous experts.
/// </summary>
public sealed class ModelConfig
{
    // Vocabulary / Tokenizer
    public int VocabSize { get; init; } = 32_000;
    public int MaxSequenceLength { get; init; } = 2048;

    // Shared backbone
    public int HiddenDim { get; init; } = 768;
    public int NumHeads { get; init; } = 12;
    public int NumLayers { get; init; } = 6;
    public double DropoutRate { get; init; } = 0.1;

    // Feed-forward (used inside experts)
    public int FeedForwardDim { get; init; } = 3072; // 4 * HiddenDim

    // MoE routing
    public int NumExperts { get; init; } = 8;
    public int TopK { get; init; } = 2;
    public double LoadBalanceLossWeight { get; init; } = 0.01;
    public double RouterZLossWeight { get; init; } = 0.001;

    // LoRA defaults
    public int LoraRank { get; init; } = 16;
    public double LoraAlpha { get; init; } = 32.0;
    public double LoraDropout { get; init; } = 0.05;

    /// <summary>
    /// Per-expert depth overrides. Index = expert id, value = number of FFN sub-layers.
    /// If null or missing, defaults to 2 sub-layers.
    /// </summary>
    public int[]? ExpertDepths { get; set; }

    public int GetExpertDepth(int expertIndex)
    {
        if (ExpertDepths is not null && expertIndex < ExpertDepths.Length)
            return ExpertDepths[expertIndex];
        return 2; // default depth
    }
}
