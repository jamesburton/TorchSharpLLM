namespace TorchSharpLLM.Training;

/// <summary>
/// Configuration for training runs.
/// </summary>
public sealed class TrainingConfig
{
    public double LearningRate { get; init; } = 3e-4;
    public double WeightDecay { get; init; } = 0.01;
    public double Beta1 { get; init; } = 0.9;
    public double Beta2 { get; init; } = 0.95;
    public double MaxGradNorm { get; init; } = 1.0;
    public int BatchSize { get; init; } = 4;
    public int SequenceLength { get; init; } = 512;
    public int NumEpochs { get; init; } = 1;
    public int LogEveryNSteps { get; init; } = 10;
    public int SaveEveryNSteps { get; init; } = 500;
    public string CheckpointDir { get; init; } = "checkpoints";

    /// <summary>
    /// If set, only these expert indices will be trainable (per-user training).
    /// Null = train everything.
    /// </summary>
    public int[]? TrainableExpertIndices { get; init; }

    /// <summary>
    /// If true, uses LoRA for the trainable experts instead of full fine-tuning.
    /// </summary>
    public bool UseLoRA { get; init; } = false;
}
