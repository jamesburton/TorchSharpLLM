using TorchSharp;
using static TorchSharp.torch;

namespace TorchSharpLLM.Models;

/// <summary>
/// Manages dynamic expert lifecycle across all MoE layers:
///   - Add new experts (with configurable depth)
///   - Remove experts by ID
///   - Grow existing experts (add sub-layers)
///   - Apply/merge/unmerge LoRA on specific experts
///   - Freeze/unfreeze experts for per-user training
///   - Save/load individual expert checkpoints
/// </summary>
public sealed class ExpertManager
{
    private readonly MoETransformerModel _model;
    private readonly ModelConfig _config;

    public ExpertManager(MoETransformerModel model, ModelConfig config)
    {
        _model = model;
        _config = config;
    }

    /// <summary>
    /// Add a new expert to every MoE layer. Returns the new expert's ID.
    /// </summary>
    public int AddExpert(int depth = 2, double? dropoutRate = null)
    {
        double dropout = dropoutRate ?? _config.DropoutRate;
        int newId = -1;

        foreach (var moeLayer in _model.MoELayers)
        {
            int currentCount = (int)moeLayer.Experts.Count;
            newId = currentCount; // Same ID across all layers

            var expert = new Expert(newId, _config.HiddenDim, _config.FeedForwardDim, depth, dropout);
            expert = expert.to(_model.Device);
            moeLayer.Experts.append(expert);

            // Resize the router gate to include the new expert
            moeLayer.Router.ResizeExperts(currentCount + 1, _config.HiddenDim);
        }

        Console.WriteLine($"[ExpertManager] Added expert {newId} with depth {depth} across {_model.MoELayers.Count} layers");
        return newId;
    }

    /// <summary>
    /// Remove an expert by index from every MoE layer.
    /// Shifts subsequent experts down but does NOT re-number their internal IDs.
    /// </summary>
    public void RemoveExpert(int expertIndex)
    {
        foreach (var moeLayer in _model.MoELayers)
        {
            int count = (int)moeLayer.Experts.Count;
            if (expertIndex < 0 || expertIndex >= count)
                throw new ArgumentOutOfRangeException(nameof(expertIndex),
                    $"Expert index {expertIndex} out of range [0, {count})");

            // Build a new list without the removed expert
            var remaining = new List<Expert>();
            for (int i = 0; i < count; i++)
            {
                if (i == expertIndex)
                {
                    moeLayer.Experts[i].Dispose();
                    continue;
                }
                remaining.Add(moeLayer.Experts[i]);
            }

            // Re-register. ModuleList doesn't support removal, so we clear and re-add.
            while (moeLayer.Experts.Count > 0)
            {
                // Remove last to avoid index shifting issues on internal list
                // We already hold references in 'remaining', so just need to repopulate
                break;
            }

            // Recreate the expert list in the MoE layer
            // We'll use the model's RebuildExperts helper
            _model.RebuildExpertsForLayer(moeLayer, remaining);

            moeLayer.Router.ResizeExperts(remaining.Count, _config.HiddenDim);
        }

        Console.WriteLine($"[ExpertManager] Removed expert {expertIndex} across all layers");
    }

    /// <summary>
    /// Add sub-layers to an existing expert to increase its capacity.
    /// </summary>
    public void GrowExpert(int expertIndex, int additionalLayers)
    {
        foreach (var moeLayer in _model.MoELayers)
        {
            var expert = moeLayer.Experts[expertIndex];
            int oldDepth = expert.Depth;
            expert.AddSubLayers(additionalLayers, _config.DropoutRate);
            Console.WriteLine($"[ExpertManager] Grew expert {expertIndex}: depth {oldDepth} → {expert.Depth}");
        }
    }

    /// <summary>
    /// Apply LoRA adapters to a specific expert across all layers.
    /// </summary>
    public void ApplyLoRA(int expertIndex, int? rank = null, double? alpha = null, double? dropout = null)
    {
        int r = rank ?? _config.LoraRank;
        double a = alpha ?? _config.LoraAlpha;
        double d = dropout ?? _config.LoraDropout;

        foreach (var moeLayer in _model.MoELayers)
        {
            moeLayer.Experts[expertIndex].ApplyLoRA(r, a, d);
        }
        Console.WriteLine($"[ExpertManager] Applied LoRA (rank={r}) to expert {expertIndex}");
    }

    /// <summary>
    /// Merge LoRA weights into base weights for an expert (inference optimization).
    /// </summary>
    public void MergeLoRA(int expertIndex)
    {
        foreach (var moeLayer in _model.MoELayers)
            moeLayer.Experts[expertIndex].MergeLoRA();
        Console.WriteLine($"[ExpertManager] Merged LoRA for expert {expertIndex}");
    }

    /// <summary>
    /// Un-merge LoRA weights for an expert (resume training).
    /// </summary>
    public void UnmergeLoRA(int expertIndex)
    {
        foreach (var moeLayer in _model.MoELayers)
            moeLayer.Experts[expertIndex].UnmergeLoRA();
        Console.WriteLine($"[ExpertManager] Unmerged LoRA for expert {expertIndex}");
    }

    /// <summary>
    /// Freeze all model parameters, then selectively unfreeze specific experts.
    /// This enables per-user expert training while keeping the shared backbone frozen.
    /// </summary>
    public void SetupSelectiveTraining(params int[] trainableExpertIndices)
    {
        // Freeze everything
        foreach (var p in _model.parameters())
            p.requires_grad_(false);

        // Unfreeze selected experts
        foreach (int idx in trainableExpertIndices)
        {
            foreach (var moeLayer in _model.MoELayers)
                moeLayer.Experts[idx].SetTrainable(true);
        }

        Console.WriteLine($"[ExpertManager] Selective training: experts [{string.Join(", ", trainableExpertIndices)}] are trainable, everything else frozen");
    }

    /// <summary>
    /// Setup LoRA-only training: freeze base weights in specified experts,
    /// apply LoRA if not already applied, and only train the LoRA parameters.
    /// </summary>
    public void SetupLoRATraining(params int[] expertIndices)
    {
        // Freeze everything first
        foreach (var p in _model.parameters())
            p.requires_grad_(false);

        foreach (int idx in expertIndices)
        {
            // Apply LoRA if not already
            ApplyLoRA(idx);

            foreach (var moeLayer in _model.MoELayers)
                moeLayer.Experts[idx].FreezeBaseKeepLoRA();
        }

        Console.WriteLine($"[ExpertManager] LoRA training: experts [{string.Join(", ", expertIndices)}] LoRA params trainable");
    }

    /// <summary>
    /// Unfreeze all parameters for full retraining.
    /// </summary>
    public void SetupFullRetraining()
    {
        foreach (var p in _model.parameters())
            p.requires_grad_(true);
        Console.WriteLine("[ExpertManager] Full retraining: all parameters unfrozen");
    }

    /// <summary>
    /// Save a single expert's weights to disk (across all layers).
    /// </summary>
    public void SaveExpert(int expertIndex, string directory)
    {
        Directory.CreateDirectory(directory);
        int layerIdx = 0;
        foreach (var moeLayer in _model.MoELayers)
        {
            string path = Path.Combine(directory, $"expert_{expertIndex}_layer_{layerIdx}.pt");
            moeLayer.Experts[expertIndex].save(path);
            layerIdx++;
        }
        Console.WriteLine($"[ExpertManager] Saved expert {expertIndex} to {directory}");
    }

    /// <summary>
    /// Load a single expert's weights from disk (across all layers).
    /// </summary>
    public void LoadExpert(int expertIndex, string directory)
    {
        int layerIdx = 0;
        foreach (var moeLayer in _model.MoELayers)
        {
            string path = Path.Combine(directory, $"expert_{expertIndex}_layer_{layerIdx}.pt");
            if (File.Exists(path))
            {
                moeLayer.Experts[expertIndex].load(path);
            }
            layerIdx++;
        }
        Console.WriteLine($"[ExpertManager] Loaded expert {expertIndex} from {directory}");
    }

    /// <summary>
    /// Get parameter count statistics per expert.
    /// </summary>
    public Dictionary<int, long> GetExpertParameterCounts()
    {
        var counts = new Dictionary<int, long>();

        // Use first MoE layer as representative (experts are mirrored across layers)
        var firstLayer = _model.MoELayers.First();
        for (int i = 0; i < (int)firstLayer.Experts.Count; i++)
        {
            long totalParams = firstLayer.Experts[i].parameters().Sum(p => p.numel());
            // Multiply by number of MoE layers since each expert exists in every layer
            counts[i] = totalParams * _model.MoELayers.Count;
        }

        return counts;
    }
}
