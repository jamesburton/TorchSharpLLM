using TorchSharp;
using TorchSharpLLM.Models;
using TorchSharpLLM.Training;
using static TorchSharp.torch;

Console.WriteLine("╔══════════════════════════════════════════════════════════╗");
Console.WriteLine("║       TorchSharp MoE Transformer — Demo                 ║");
Console.WriteLine("╚══════════════════════════════════════════════════════════╝");
Console.WriteLine();

// ────────────────────────────────────────────────────────────────────────────
// 1. Configure the model: ~1.5GB with 8 heterogeneous experts
// ────────────────────────────────────────────────────────────────────────────
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

    // Heterogeneous expert depths: some shallow, some deep
    ExpertDepths = new[] { 1, 2, 2, 3, 2, 2, 3, 1 },

    // LoRA defaults
    LoraRank = 16,
    LoraAlpha = 32.0,
    LoraDropout = 0.05
};

Console.WriteLine("Building model...");
var model = new MoETransformerModel(config);
var expertManager = new ExpertManager(model, config);

// ────────────────────────────────────────────────────────────────────────────
// 2. Report model size
// ────────────────────────────────────────────────────────────────────────────
var sizeInfo = model.GetSizeInfo();
Console.WriteLine(sizeInfo);
Console.WriteLine();

// Per-expert parameter counts
Console.WriteLine("Per-expert parameter counts:");
var expertCounts = expertManager.GetExpertParameterCounts();
foreach (var (id, count) in expertCounts)
{
    double mb = count * 4.0 / (1024 * 1024);
    Console.WriteLine($"  Expert {id}: {count:N0} params ({mb:F1} MB) — depth {config.GetExpertDepth(id)}");
}
Console.WriteLine();

// ────────────────────────────────────────────────────────────────────────────
// 3. Demo: Forward pass
// ────────────────────────────────────────────────────────────────────────────
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

// ────────────────────────────────────────────────────────────────────────────
// 4. Demo: Dynamic expert management
// ────────────────────────────────────────────────────────────────────────────
Console.WriteLine("── Dynamic expert management demo ──");

// Add a new expert (9th, deep)
int newId = expertManager.AddExpert(depth: 4);
Console.WriteLine($"Added expert {newId} with depth 4");

// Grow an existing expert
expertManager.GrowExpert(expertIndex: 0, additionalLayers: 2);

// Report updated counts
var updatedCounts = expertManager.GetExpertParameterCounts();
Console.WriteLine($"Now have {updatedCounts.Count} experts");
Console.WriteLine();

// Remove the extra expert
expertManager.RemoveExpert(newId);
Console.WriteLine($"Removed expert {newId}, back to {expertManager.GetExpertParameterCounts().Count} experts");
Console.WriteLine();

// ────────────────────────────────────────────────────────────────────────────
// 5. Demo: LoRA on a specific expert
// ────────────────────────────────────────────────────────────────────────────
Console.WriteLine("── LoRA demo ──");
expertManager.ApplyLoRA(expertIndex: 3, rank: 16, alpha: 32);

var sizeAfterLoRA = model.GetSizeInfo();
Console.WriteLine($"Params after LoRA on expert 3: {sizeAfterLoRA.TotalParameters:N0} " +
                  $"(+{sizeAfterLoRA.TotalParameters - sizeInfo.TotalParameters:N0} from LoRA)");

// Merge for inference
expertManager.MergeLoRA(3);
Console.WriteLine("Merged LoRA weights into expert 3 base weights");

// Unmerge to resume training
expertManager.UnmergeLoRA(3);
Console.WriteLine("Unmerged LoRA weights for continued training");
Console.WriteLine();

// ────────────────────────────────────────────────────────────────────────────
// 6. Demo: Per-user selective training
// ────────────────────────────────────────────────────────────────────────────
Console.WriteLine("── Per-user selective training demo ──");

// Scenario: User A only trains experts 0 and 1
expertManager.SetupSelectiveTraining(0, 1);
var selectiveInfo = model.GetSizeInfo();
Console.WriteLine($"Trainable params (experts 0,1 only): {selectiveInfo.TrainableParameters:N0} " +
                  $"/ {selectiveInfo.TotalParameters:N0} total");

// Scenario: User B uses LoRA on expert 5
expertManager.SetupLoRATraining(5);
var loraInfo = model.GetSizeInfo();
Console.WriteLine($"Trainable params (LoRA expert 5):    {loraInfo.TrainableParameters:N0} " +
                  $"/ {loraInfo.TotalParameters:N0} total");

// Full retraining mode
expertManager.SetupFullRetraining();
Console.WriteLine();

// ────────────────────────────────────────────────────────────────────────────
// 7. Demo: Training step (synthetic data)
// ────────────────────────────────────────────────────────────────────────────
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

// ────────────────────────────────────────────────────────────────────────────
// 8. Demo: Save/load individual experts
// ────────────────────────────────────────────────────────────────────────────
Console.WriteLine("── Expert save/load demo ──");
string exportDir = Path.Combine(Path.GetTempPath(), "torchsharp_moe_experts");
expertManager.SaveExpert(0, Path.Combine(exportDir, "expert_0"));
expertManager.SaveExpert(3, Path.Combine(exportDir, "expert_3"));
Console.WriteLine($"Experts saved to {exportDir}");
Console.WriteLine();

// ────────────────────────────────────────────────────────────────────────────
Console.WriteLine("╔══════════════════════════════════════════════════════════╗");
Console.WriteLine("║                    Demo complete!                       ║");
Console.WriteLine("╚══════════════════════════════════════════════════════════╝");
