using TorchSharp;
using TorchSharpLLM.Data;
using TorchSharpLLM.Models;
using TorchSharpLLM.Training;
using static TorchSharp.torch;

// ────────────────────────────────────────────────────────────────────────────
// Parse command-line: "demo" (default) or "train [options]"
// ────────────────────────────────────────────────────────────────────────────
var mode = args.Length > 0 ? args[0].ToLowerInvariant() : "demo";

// Shared model config (~1.5GB with 8 heterogeneous experts)
var config = new ModelConfig
{
    VocabSize = 32_000,
    MaxSequenceLength = 2048,
    HiddenDim = 768,
    NumHeads = 12,
    NumLayers = 6,
    FeedForwardDim = 3072,
    DropoutRate = 0.1,
    NumExperts = 8,
    TopK = 2,
    LoadBalanceLossWeight = 0.01,
    RouterZLossWeight = 0.001,
    ExpertDepths = new[] { 1, 2, 2, 3, 2, 2, 3, 1 },
    LoraRank = 16,
    LoraAlpha = 32.0,
    LoraDropout = 0.05
};

switch (mode)
{
    case "train":
        RunTraining(args, config);
        break;
    case "demo":
    default:
        RunDemo(config);
        break;
}

// ════════════════════════════════════════════════════════════════════════════
// TRAINING MODE
// ════════════════════════════════════════════════════════════════════════════
static void RunTraining(string[] args, ModelConfig config)
{
    Console.WriteLine("╔══════════════════════════════════════════════════════════╗");
    Console.WriteLine("║       TorchSharp MoE Transformer — Training             ║");
    Console.WriteLine("╚══════════════════════════════════════════════════════════╝");
    Console.WriteLine();

    // Parse training arguments
    string dataDir = GetArg(args, "--data-dir") ?? "data";
    double lr = double.Parse(GetArg(args, "--lr") ?? "3e-4");
    int batchSize = int.Parse(GetArg(args, "--batch-size") ?? "4");
    int seqLen = int.Parse(GetArg(args, "--seq-len") ?? "512");
    int epochs = int.Parse(GetArg(args, "--epochs") ?? "1");
    int? maxSteps = GetArg(args, "--max-steps") is string ms ? int.Parse(ms) : null;
    int logEvery = int.Parse(GetArg(args, "--log-every") ?? "10");
    int saveEvery = int.Parse(GetArg(args, "--save-every") ?? "500");
    string checkpointDir = GetArg(args, "--checkpoint-dir") ?? "checkpoints";
    bool useLora = HasFlag(args, "--lora");
    string mixing = GetArg(args, "--mixing") ?? "Proportional";
    int? targetDomain = GetArg(args, "--domain") is string td ? int.Parse(td) : null;

    int[]? expertIndices = null;
    string? expertsStr = GetArg(args, "--experts");
    if (expertsStr is not null)
    {
        expertIndices = expertsStr.Split(',', ' ')
            .Where(s => !string.IsNullOrWhiteSpace(s))
            .Select(int.Parse)
            .ToArray();
    }

    // Also collect trailing expert indices after --experts flag
    if (expertIndices is null)
    {
        int expIdx = Array.IndexOf(args, "--experts");
        if (expIdx >= 0)
        {
            var indices = new List<int>();
            for (int i = expIdx + 1; i < args.Length; i++)
            {
                if (args[i].StartsWith("--")) break;
                if (int.TryParse(args[i], out int idx))
                    indices.Add(idx);
            }
            if (indices.Count > 0)
                expertIndices = indices.ToArray();
        }
    }

    var trainingConfig = new TrainingConfig
    {
        LearningRate = lr,
        BatchSize = batchSize,
        SequenceLength = seqLen,
        NumEpochs = epochs,
        MaxSteps = maxSteps,
        LogEveryNSteps = logEvery,
        SaveEveryNSteps = saveEvery,
        CheckpointDir = checkpointDir,
        UseLoRA = useLora,
        TrainableExpertIndices = expertIndices,
        DataDir = dataDir,
        MixingStrategy = mixing,
    };

    // Print config
    Console.WriteLine("Training configuration:");
    Console.WriteLine($"  Data dir:        {dataDir}");
    Console.WriteLine($"  Learning rate:   {lr}");
    Console.WriteLine($"  Batch size:      {batchSize}");
    Console.WriteLine($"  Sequence length: {seqLen}");
    Console.WriteLine($"  Epochs:          {epochs}");
    Console.WriteLine($"  Max steps:       {maxSteps?.ToString() ?? "auto"}");
    Console.WriteLine($"  Mixing:          {mixing}");
    Console.WriteLine($"  LoRA:            {useLora}");
    Console.WriteLine($"  Experts:         {(expertIndices is null ? "all" : string.Join(", ", expertIndices))}");
    Console.WriteLine($"  Target domain:   {(targetDomain.HasValue ? targetDomain.Value.ToString() : "all")}");
    Console.WriteLine($"  Checkpoint dir:  {checkpointDir}");
    Console.WriteLine();

    // Build model
    Console.WriteLine("Building model...");
    var model = new MoETransformerModel(config);
    var expertManager = new ExpertManager(model, config);

    var sizeInfo = model.GetSizeInfo();
    Console.WriteLine(sizeInfo);
    Console.WriteLine();

    // Load data
    Console.WriteLine("Loading tokenized datasets...");
    var strategy = Enum.Parse<MixingStrategy>(mixing, ignoreCase: true);
    using var dataLoader = new MoEDataLoader(batchSize, seqLen, model.Device, strategy, seed: 42);
    dataLoader.AutoLoadFromDirectory(dataDir);

    if (dataLoader.DomainCount == 0)
    {
        Console.Error.WriteLine("ERROR: No datasets found. Run scripts/setup_data.sh first.");
        Environment.Exit(1);
    }
    Console.WriteLine();

    // Setup trainer
    var trainer = new Trainer(model, expertManager, trainingConfig, config);
    trainer.Prepare();

    // Train
    if (targetDomain.HasValue)
    {
        trainer.TrainOnDomain(dataLoader, targetDomain.Value, maxSteps ?? 1000);
    }
    else
    {
        trainer.TrainOnDatasets(dataLoader, maxSteps);
    }

    // Save final checkpoint
    string finalPath = Path.Combine(checkpointDir, "final_model.pt");
    Directory.CreateDirectory(checkpointDir);
    model.save(finalPath);
    Console.WriteLine($"\nFinal model saved to {finalPath}");
}

// ════════════════════════════════════════════════════════════════════════════
// DEMO MODE (original)
// ════════════════════════════════════════════════════════════════════════════
static void RunDemo(ModelConfig config)
{
    Console.WriteLine("╔══════════════════════════════════════════════════════════╗");
    Console.WriteLine("║       TorchSharp MoE Transformer — Demo                 ║");
    Console.WriteLine("╚══════════════════════════════════════════════════════════╝");
    Console.WriteLine();

    Console.WriteLine("Building model...");
    var model = new MoETransformerModel(config);
    var expertManager = new ExpertManager(model, config);

    // Report model size
    var sizeInfo = model.GetSizeInfo();
    Console.WriteLine(sizeInfo);
    Console.WriteLine();

    Console.WriteLine("Per-expert parameter counts:");
    var expertCounts = expertManager.GetExpertParameterCounts();
    foreach (var (id, count) in expertCounts)
    {
        double mb = count * 4.0 / (1024 * 1024);
        Console.WriteLine($"  Expert {id}: {count:N0} params ({mb:F1} MB) — depth {config.GetExpertDepth(id)}");
    }
    Console.WriteLine();

    // Forward pass
    Console.WriteLine("── Forward pass demo ──");
    using (torch.no_grad())
    {
        using var input = torch.randint(config.VocabSize, new long[] { 1, 64 });
        var output = model.forward(input);
        Console.WriteLine($"Input shape:  [{string.Join(", ", input.shape)}]");
        Console.WriteLine($"Output shape: [{string.Join(", ", output.Logits.shape)}]");
        Console.WriteLine($"Aux loss:     {output.AuxLoss.item<float>():F6}");
        output.Logits.Dispose();
        output.AuxLoss.Dispose();
    }
    Console.WriteLine();

    // Dynamic expert management
    Console.WriteLine("── Dynamic expert management demo ──");
    int newId = expertManager.AddExpert(depth: 4);
    Console.WriteLine($"Added expert {newId} with depth 4");
    expertManager.GrowExpert(expertIndex: 0, additionalLayers: 2);
    var updatedCounts = expertManager.GetExpertParameterCounts();
    Console.WriteLine($"Now have {updatedCounts.Count} experts");
    Console.WriteLine();
    expertManager.RemoveExpert(newId);
    Console.WriteLine($"Removed expert {newId}, back to {expertManager.GetExpertParameterCounts().Count} experts");
    Console.WriteLine();

    // LoRA
    Console.WriteLine("── LoRA demo ──");
    expertManager.ApplyLoRA(expertIndex: 3, rank: 16, alpha: 32);
    var sizeAfterLoRA = model.GetSizeInfo();
    Console.WriteLine($"Params after LoRA on expert 3: {sizeAfterLoRA.TotalParameters:N0} " +
                      $"(+{sizeAfterLoRA.TotalParameters - sizeInfo.TotalParameters:N0} from LoRA)");
    expertManager.MergeLoRA(3);
    Console.WriteLine("Merged LoRA weights into expert 3 base weights");
    expertManager.UnmergeLoRA(3);
    Console.WriteLine("Unmerged LoRA weights for continued training");
    Console.WriteLine();

    // Per-user selective training
    Console.WriteLine("── Per-user selective training demo ──");
    expertManager.SetupSelectiveTraining(0, 1);
    var selectiveInfo = model.GetSizeInfo();
    Console.WriteLine($"Trainable params (experts 0,1 only): {selectiveInfo.TrainableParameters:N0} " +
                      $"/ {selectiveInfo.TotalParameters:N0} total");
    expertManager.SetupLoRATraining(5);
    var loraInfo = model.GetSizeInfo();
    Console.WriteLine($"Trainable params (LoRA expert 5):    {loraInfo.TrainableParameters:N0} " +
                      $"/ {loraInfo.TotalParameters:N0} total");
    expertManager.SetupFullRetraining();
    Console.WriteLine();

    // Synthetic training
    Console.WriteLine("── Training demo (5 steps on synthetic data) ──");
    var trainingConfig = new TrainingConfig
    {
        LearningRate = 1e-4,
        BatchSize = 2,
        SequenceLength = 64,
        LogEveryNSteps = 1
    };
    var trainer = new Trainer(model, expertManager, trainingConfig, config);
    trainer.Prepare();
    trainer.TrainOnSyntheticData(numSteps: 5);
    Console.WriteLine();

    // Save/load
    Console.WriteLine("── Expert save/load demo ──");
    string exportDir = Path.Combine(Path.GetTempPath(), "torchsharp_moe_experts");
    expertManager.SaveExpert(0, Path.Combine(exportDir, "expert_0"));
    expertManager.SaveExpert(3, Path.Combine(exportDir, "expert_3"));
    Console.WriteLine($"Experts saved to {exportDir}");
    Console.WriteLine();

    Console.WriteLine("╔══════════════════════════════════════════════════════════╗");
    Console.WriteLine("║                    Demo complete!                       ║");
    Console.WriteLine("╚══════════════════════════════════════════════════════════╝");
}

// ════════════════════════════════════════════════════════════════════════════
// Argument helpers
// ════════════════════════════════════════════════════════════════════════════
static string? GetArg(string[] args, string flag)
{
    int idx = Array.IndexOf(args, flag);
    if (idx >= 0 && idx + 1 < args.Length && !args[idx + 1].StartsWith("--"))
        return args[idx + 1];
    return null;
}

static bool HasFlag(string[] args, string flag) => args.Contains(flag);
