using TorchSharp;
using TorchSharpLLM.Models;
using static TorchSharp.torch;

namespace TorchSharpLLM.Training;

/// <summary>
/// Training harness supporting:
///   - Full model retraining
///   - Selective expert training (per-user)
///   - LoRA-based expert fine-tuning
///   - Gradient clipping, AdamW optimizer
///   - Auxiliary load-balancing loss integration
/// </summary>
public sealed class Trainer
{
    private readonly MoETransformerModel _model;
    private readonly ExpertManager _expertManager;
    private readonly TrainingConfig _config;
    private readonly ModelConfig _modelConfig;
    private torch.optim.Optimizer? _optimizer;

    public Trainer(MoETransformerModel model, ExpertManager expertManager,
                   TrainingConfig trainingConfig, ModelConfig modelConfig)
    {
        _model = model;
        _expertManager = expertManager;
        _config = trainingConfig;
        _modelConfig = modelConfig;
    }

    /// <summary>
    /// Prepare the model and optimizer for a training run.
    /// </summary>
    public void Prepare()
    {
        // Configure which parameters are trainable
        if (_config.TrainableExpertIndices is not null)
        {
            if (_config.UseLoRA)
                _expertManager.SetupLoRATraining(_config.TrainableExpertIndices);
            else
                _expertManager.SetupSelectiveTraining(_config.TrainableExpertIndices);
        }
        else
        {
            _expertManager.SetupFullRetraining();
        }

        // Create optimizer over trainable parameters only
        var trainableParams = _model.parameters()
            .Where(p => p.requires_grad)
            .ToList();

        Console.WriteLine($"[Trainer] Trainable parameters: {trainableParams.Sum(p => p.numel()):N0}");

        _optimizer = torch.optim.AdamW(
            trainableParams,
            lr: _config.LearningRate,
            weight_decay: _config.WeightDecay,
            beta1: _config.Beta1,
            beta2: _config.Beta2);
    }

    /// <summary>
    /// Execute a single training step on a batch of token IDs.
    /// Returns (total_loss, lm_loss, aux_loss).
    /// </summary>
    public (double totalLoss, double lmLoss, double auxLoss) TrainStep(Tensor inputIds, Tensor targetIds)
    {
        if (_optimizer is null)
            throw new InvalidOperationException("Call Prepare() before training.");

        _model.train();
        _optimizer.zero_grad();

        // Forward pass
        var output = _model.forward(inputIds);

        // Language modeling loss: cross-entropy over vocabulary
        // Reshape logits: [batch * seqLen, vocabSize], targets: [batch * seqLen]
        long batchSeq = output.Logits.shape[0] * output.Logits.shape[1];
        int vocabSize = (int)output.Logits.shape[2];

        using var flatLogits = output.Logits.view(batchSeq, vocabSize);
        using var flatTargets = targetIds.view(batchSeq);

        using var lmLoss = torch.nn.functional.cross_entropy(flatLogits, flatTargets);

        // Total loss = LM loss + auxiliary router loss
        using var totalLoss = lmLoss + output.AuxLoss;

        // Backward pass
        totalLoss.backward();

        // Gradient clipping
        torch.nn.utils.clip_grad_norm_(_model.parameters(), _config.MaxGradNorm);

        // Optimizer step
        _optimizer.step();

        double totalVal = totalLoss.item<float>();
        double lmVal = lmLoss.item<float>();
        double auxVal = output.AuxLoss.item<float>();

        output.Logits.Dispose();
        output.AuxLoss.Dispose();

        return (totalVal, lmVal, auxVal);
    }

    /// <summary>
    /// Run a full training loop over synthetic data (for demonstration).
    /// In production, replace with a real data loader.
    /// </summary>
    public void TrainOnSyntheticData(int numSteps)
    {
        Console.WriteLine($"[Trainer] Starting synthetic training for {numSteps} steps...");
        Console.WriteLine($"[Trainer] Batch size: {_config.BatchSize}, Seq length: {_config.SequenceLength}");

        for (int step = 1; step <= numSteps; step++)
        {
            // Generate random token IDs as synthetic data
            using var inputIds = torch.randint(_modelConfig.VocabSize,
                new long[] { _config.BatchSize, _config.SequenceLength },
                device: _model.Device);

            // Targets = shifted input (next token prediction)
            using var targetIds = torch.randint(_modelConfig.VocabSize,
                new long[] { _config.BatchSize, _config.SequenceLength },
                device: _model.Device);

            var (totalLoss, lmLoss, auxLoss) = TrainStep(inputIds, targetIds);

            if (step % _config.LogEveryNSteps == 0 || step == 1)
            {
                Console.WriteLine($"  Step {step}/{numSteps} | Total: {totalLoss:F4} | LM: {lmLoss:F4} | Aux: {auxLoss:F6}");
            }

            if (step % _config.SaveEveryNSteps == 0)
            {
                string path = Path.Combine(_config.CheckpointDir, $"checkpoint_step_{step}.pt");
                Directory.CreateDirectory(_config.CheckpointDir);
                _model.save(path);
                Console.WriteLine($"  Saved checkpoint: {path}");
            }
        }

        Console.WriteLine("[Trainer] Training complete.");
    }
}
