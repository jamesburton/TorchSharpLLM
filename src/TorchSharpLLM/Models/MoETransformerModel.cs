using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace TorchSharpLLM.Models;

/// <summary>
/// Full MoE Transformer model:
///   Token Embedding → [Attention + MoE FFN] × N layers → RMSNorm → LM Head
///
/// Architecture:
///   - Shared backbone: token embeddings + multi-head attention (shared across all tokens)
///   - Per-layer MoE: router selects top-k experts from a pool of heterogeneous FFN experts
///   - Each expert can have different depth (number of FFN sub-layers)
///   - LoRA can be applied to any expert for parameter-efficient fine-tuning
/// </summary>
public sealed class MoETransformerModel : Module<Tensor, MoETransformerModel.ModelOutput>
{
    private readonly Embedding _tokenEmbedding;
    private readonly ModuleList<MultiHeadAttention> _attentionLayers;
    private readonly ModuleList<MoELayer> _moeLayers;
    private readonly RMSNorm _finalNorm;
    private readonly Linear _lmHead;
    private readonly Dropout _dropout;
    private readonly ModelConfig _config;

    // RoPE tables (not trainable, registered as buffers)
    private Tensor _ropeCos;
    private Tensor _ropeSin;

    public IReadOnlyList<MoELayer> MoELayers
    {
        get
        {
            var list = new List<MoELayer>();
            for (int i = 0; i < _moeLayers.Count; i++)
                list.Add(_moeLayers[i]);
            return list;
        }
    }

    public Device Device { get; private set; }

    public MoETransformerModel(ModelConfig config) : base("MoETransformerModel")
    {
        _config = config;
        Device = torch.CPU;

        // Token embedding (shared, no positional embedding — we use RoPE)
        _tokenEmbedding = Embedding(config.VocabSize, config.HiddenDim);
        _dropout = Dropout(config.DropoutRate);

        // Shared attention layers + per-layer MoE
        var attnLayers = new List<MultiHeadAttention>();
        var moeLayers = new List<MoELayer>();

        for (int i = 0; i < config.NumLayers; i++)
        {
            attnLayers.Add(new MultiHeadAttention(config.HiddenDim, config.NumHeads, config.DropoutRate));
            moeLayers.Add(new MoELayer(config, i));
        }

        _attentionLayers = new ModuleList<MultiHeadAttention>(attnLayers.ToArray());
        _moeLayers = new ModuleList<MoELayer>(moeLayers.ToArray());

        // Final normalization and language model head
        _finalNorm = new RMSNorm(config.HiddenDim);
        _lmHead = Linear(config.HiddenDim, config.VocabSize, hasBias: false);

        // Tie embedding and LM head weights
        _lmHead.weight = _tokenEmbedding.weight;

        // Pre-compute RoPE tables
        var (cos, sin) = RotaryPositionalEmbedding.Precompute(
            config.MaxSequenceLength,
            config.HiddenDim / config.NumHeads);
        _ropeCos = cos;
        _ropeSin = sin;

        RegisterComponents();
    }

    public override ModelOutput forward(Tensor inputIds)
    {
        // inputIds: [batch, seqLen] of token IDs
        using var embedded = _tokenEmbedding.forward(inputIds);
        var x = _dropout.forward(embedded);

        var totalAuxLoss = torch.tensor(0.0f, device: x.device);

        // Move RoPE tables to device if needed
        if (_ropeCos.device != x.device)
        {
            _ropeCos = _ropeCos.to(x.device);
            _ropeSin = _ropeSin.to(x.device);
        }

        for (int i = 0; i < _config.NumLayers; i++)
        {
            // Self-attention (shared backbone)
            var afterAttn = _attentionLayers[i].forward(x, _ropeCos, _ropeSin);
            if (!ReferenceEquals(x, embedded))
                x.Dispose();
            x = afterAttn;

            // MoE feed-forward
            var (moeOut, auxLoss) = _moeLayers[i].forward(x);
            x.Dispose();
            x = moeOut;

            using var oldAux = totalAuxLoss;
            totalAuxLoss = totalAuxLoss + auxLoss;
        }

        // Final norm → logits
        using var normed = _finalNorm.forward(x);
        x.Dispose();
        var logits = _lmHead.forward(normed);

        return new ModelOutput
        {
            Logits = logits,           // [batch, seqLen, vocabSize]
            AuxLoss = totalAuxLoss     // scalar
        };
    }

    /// <summary>
    /// Move model to a specific device.
    /// </summary>
    public new MoETransformerModel to(Device device)
    {
        Device = device;
        base.to(device);
        _ropeCos = _ropeCos.to(device);
        _ropeSin = _ropeSin.to(device);
        return this;
    }

    /// <summary>
    /// Helper to rebuild experts list after removal (called by ExpertManager).
    /// </summary>
    internal void RebuildExpertsForLayer(MoELayer layer, List<Expert> experts)
    {
        // Clear and repopulate the expert ModuleList
        // Since TorchSharp ModuleList doesn't support removal, we replace the contents
        // by creating a new module list.
        // We work around this by re-registering each expert.
        var currentCount = (int)layer.Experts.Count;

        // We can't truly remove from ModuleList easily in TorchSharp,
        // so we'll replace experts at existing indices and handle the count difference
        // by building a fresh list and re-registering.
        // The simplest approach: clear all, re-add remaining.
        for (int i = currentCount - 1; i >= 0; i--)
        {
            // Don't dispose — caller already handled disposal of removed ones
            // and kept references to remaining ones
        }

        // Re-append remaining experts
        var newList = new ModuleList<Expert>(experts.ToArray());
        // This is a workaround: we register the new list under the same name
        // The MoELayer will need to reference this new list.
        layer.register_module("_experts", newList);
    }

    /// <summary>
    /// Count total parameters and report size breakdown.
    /// </summary>
    public ModelSizeInfo GetSizeInfo()
    {
        long totalParams = 0;
        long trainableParams = 0;
        long embeddingParams = 0;
        long attentionParams = 0;
        long expertParams = 0;
        long routerParams = 0;
        long lmHeadParams = 0;

        foreach (var (name, param) in named_parameters())
        {
            long count = param.numel();
            totalParams += count;
            if (param.requires_grad) trainableParams += count;

            if (name.Contains("_tokenEmbedding"))
                embeddingParams += count;
            else if (name.Contains("_attentionLayers"))
                attentionParams += count;
            else if (name.Contains("_experts"))
                expertParams += count;
            else if (name.Contains("_router") || name.Contains("_gate"))
                routerParams += count;
            else if (name.Contains("_lmHead"))
                lmHeadParams += count;
        }

        return new ModelSizeInfo
        {
            TotalParameters = totalParams,
            TrainableParameters = trainableParams,
            EmbeddingParameters = embeddingParams,
            AttentionParameters = attentionParams,
            ExpertParameters = expertParams,
            RouterParameters = routerParams,
            LMHeadParameters = lmHeadParams,
            TotalSizeBytes = totalParams * 4, // float32
            TotalSizeMB = totalParams * 4.0 / (1024 * 1024),
            TotalSizeGB = totalParams * 4.0 / (1024 * 1024 * 1024)
        };
    }

    public sealed class ModelOutput
    {
        public required Tensor Logits { get; init; }
        public required Tensor AuxLoss { get; init; }
    }

    public sealed class ModelSizeInfo
    {
        public long TotalParameters { get; init; }
        public long TrainableParameters { get; init; }
        public long EmbeddingParameters { get; init; }
        public long AttentionParameters { get; init; }
        public long ExpertParameters { get; init; }
        public long RouterParameters { get; init; }
        public long LMHeadParameters { get; init; }
        public long TotalSizeBytes { get; init; }
        public double TotalSizeMB { get; init; }
        public double TotalSizeGB { get; init; }

        public override string ToString() =>
            $"""
            === MoE Transformer Model Size ===
            Total parameters:      {TotalParameters:N0}
            Trainable parameters:  {TrainableParameters:N0}
            ─────────────────────────────────
            Embeddings:            {EmbeddingParameters:N0}
            Attention layers:      {AttentionParameters:N0}
            Expert FFN layers:     {ExpertParameters:N0}
            Router/gating:         {RouterParameters:N0}
            LM Head (tied):        {LMHeadParameters:N0}
            ─────────────────────────────────
            Total size:            {TotalSizeMB:F1} MB ({TotalSizeGB:F2} GB)
            """;
    }
}
