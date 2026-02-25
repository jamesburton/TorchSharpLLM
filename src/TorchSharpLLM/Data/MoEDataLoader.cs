using TorchSharp;
using static TorchSharp.torch;

namespace TorchSharpLLM.Data;

/// <summary>
/// Data loader for MoE training that reads pre-tokenized binary files,
/// supports multiple domains, and provides batches as TorchSharp tensors.
///
/// Supports two mixing strategies:
///   - Proportional: sample from each domain proportional to its size
///   - RoundRobin:   cycle through domains, one batch per domain at a time
///
/// Each batch includes optional domain labels so the training loop can
/// track per-domain loss and expert utilisation.
/// </summary>
public sealed class MoEDataLoader : IDisposable
{
    private readonly List<DomainData> _domains = new();
    private readonly int _batchSize;
    private readonly int _seqLength;
    private readonly Device _device;
    private readonly MixingStrategy _strategy;
    private readonly Random _rng;

    // Position cursors per domain (for sequential reading)
    private readonly Dictionary<int, int> _cursors = new();

    public int DomainCount => _domains.Count;
    public long TotalTokens => _domains.Sum(d => (long)d.Reader.NumTokens);

    public MoEDataLoader(int batchSize, int seqLength, Device device,
                          MixingStrategy strategy = MixingStrategy.Proportional,
                          int? seed = null)
    {
        _batchSize = batchSize;
        _seqLength = seqLength;
        _device = device;
        _strategy = strategy;
        _rng = seed.HasValue ? new Random(seed.Value) : new Random();
    }

    /// <summary>
    /// Add a tokenized dataset to the loader.
    /// </summary>
    public void AddDomain(int datasetId, string tokensFilePath)
    {
        var reader = BinaryTokenReader.Load(tokensFilePath);
        var info = DatasetCatalog.ById(datasetId);

        _domains.Add(new DomainData
        {
            DatasetId = datasetId,
            Info = info,
            Reader = reader,
            Weight = 1.0,  // adjusted in FinalizeWeights
        });
        _cursors[datasetId] = 0;

        Console.WriteLine($"[DataLoader] Added domain '{info?.Name ?? $"id={datasetId}"}': " +
                          $"{reader.NumTokens:N0} tokens");
    }

    /// <summary>
    /// Scan the data directory and auto-load all available tokenized datasets.
    /// </summary>
    public void AutoLoadFromDirectory(string dataDir)
    {
        foreach (var dataset in DatasetCatalog.All)
        {
            string tokensPath = Path.Combine(dataDir, dataset.Name, "tokens.bin");
            if (File.Exists(tokensPath))
            {
                AddDomain(dataset.Id, tokensPath);
            }
        }

        if (_domains.Count == 0)
            Console.WriteLine("[DataLoader] WARNING: No tokenized datasets found. Run the Python scripts first.");

        FinalizeWeights();
    }

    /// <summary>
    /// Compute sampling weights proportional to dataset size.
    /// </summary>
    private void FinalizeWeights()
    {
        long total = TotalTokens;
        if (total == 0) return;

        foreach (var d in _domains)
            d.Weight = (double)d.Reader.NumTokens / total;

        Console.WriteLine($"[DataLoader] {_domains.Count} domains, {TotalTokens:N0} total tokens");
        foreach (var d in _domains)
            Console.WriteLine($"  {d.Info?.Name ?? "?",-20} weight={d.Weight:F3} ({d.Reader.NumTokens:N0} tokens)");
    }

    /// <summary>
    /// Get the next training batch. Returns (inputIds, targetIds, domainId).
    /// inputIds:  [batchSize, seqLength]
    /// targetIds: [batchSize, seqLength]  (shifted by 1)
    /// domainId:  which domain this batch came from (-1 for mixed)
    /// </summary>
    public TrainingBatch NextBatch()
    {
        return _strategy switch
        {
            MixingStrategy.Proportional => NextBatchProportional(),
            MixingStrategy.RoundRobin => NextBatchRoundRobin(),
            MixingStrategy.SingleDomain => throw new InvalidOperationException(
                "Use NextBatchFromDomain(domainIndex) for SingleDomain strategy"),
            _ => NextBatchProportional()
        };
    }

    /// <summary>
    /// Get a batch from a specific domain (for targeted expert training).
    /// </summary>
    public TrainingBatch NextBatchFromDomain(int domainIndex)
    {
        var domain = _domains[domainIndex];
        return CreateBatchFromDomain(domain);
    }

    private TrainingBatch NextBatchProportional()
    {
        // Each sample in the batch can come from a different domain
        var inputData = new long[_batchSize * _seqLength];
        var targetData = new long[_batchSize * _seqLength];
        int primaryDomain = -1; // mixed

        for (int b = 0; b < _batchSize; b++)
        {
            // Weighted random domain selection
            var domain = SelectWeightedDomain();
            var (input, target) = GetSequenceFromDomain(domain);

            Array.Copy(input, 0, inputData, b * _seqLength, _seqLength);
            Array.Copy(target, 0, targetData, b * _seqLength, _seqLength);
        }

        var inputTensor = torch.tensor(inputData, dtype: ScalarType.Int64)
            .view(_batchSize, _seqLength).to(_device);
        var targetTensor = torch.tensor(targetData, dtype: ScalarType.Int64)
            .view(_batchSize, _seqLength).to(_device);

        return new TrainingBatch(inputTensor, targetTensor, primaryDomain);
    }

    private int _roundRobinIndex;

    private TrainingBatch NextBatchRoundRobin()
    {
        if (_domains.Count == 0)
            throw new InvalidOperationException("No domains loaded.");

        var domain = _domains[_roundRobinIndex % _domains.Count];
        _roundRobinIndex++;
        return CreateBatchFromDomain(domain);
    }

    private TrainingBatch CreateBatchFromDomain(DomainData domain)
    {
        var inputData = new long[_batchSize * _seqLength];
        var targetData = new long[_batchSize * _seqLength];

        for (int b = 0; b < _batchSize; b++)
        {
            var (input, target) = GetSequenceFromDomain(domain);
            Array.Copy(input, 0, inputData, b * _seqLength, _seqLength);
            Array.Copy(target, 0, targetData, b * _seqLength, _seqLength);
        }

        var inputTensor = torch.tensor(inputData, dtype: ScalarType.Int64)
            .view(_batchSize, _seqLength).to(_device);
        var targetTensor = torch.tensor(targetData, dtype: ScalarType.Int64)
            .view(_batchSize, _seqLength).to(_device);

        return new TrainingBatch(inputTensor, targetTensor, domain.DatasetId);
    }

    private (long[] input, long[] target) GetSequenceFromDomain(DomainData domain)
    {
        int cursor = _cursors.GetValueOrDefault(domain.DatasetId, 0);
        int available = domain.Reader.NumTokens - _seqLength - 1;

        if (available <= 0)
        {
            // Dataset too small — pad with zeros
            return (new long[_seqLength], new long[_seqLength]);
        }

        // If cursor is past the end, wrap around (new epoch)
        if (cursor >= available)
        {
            cursor = _rng.Next(available);
        }

        // Get seqLength+1 tokens: first seqLength are input, last seqLength are target
        var slice = domain.Reader.GetSlice(cursor, _seqLength + 1);

        var input = new long[_seqLength];
        var target = new long[_seqLength];
        Array.Copy(slice, 0, input, 0, _seqLength);
        Array.Copy(slice, 1, target, 0, _seqLength);

        // Advance cursor with some randomization to avoid always reading sequentially
        _cursors[domain.DatasetId] = cursor + _seqLength + _rng.Next(0, _seqLength / 4);

        return (input, target);
    }

    private DomainData SelectWeightedDomain()
    {
        double r = _rng.NextDouble();
        double cumulative = 0;
        foreach (var d in _domains)
        {
            cumulative += d.Weight;
            if (r <= cumulative)
                return d;
        }
        return _domains[^1];
    }

    /// <summary>
    /// Get total number of batches for one epoch across all domains.
    /// </summary>
    public int BatchesPerEpoch()
    {
        long totalSeqs = _domains.Sum(d => (long)d.Reader.CountSequences(_seqLength));
        return (int)(totalSeqs / _batchSize);
    }

    /// <summary>
    /// Reset all cursors to the start (for a new epoch).
    /// </summary>
    public void ResetEpoch()
    {
        foreach (var key in _cursors.Keys.ToList())
            _cursors[key] = 0;
        _roundRobinIndex = 0;
    }

    public void Dispose()
    {
        foreach (var d in _domains)
            d.Reader.Dispose();
    }

    private class DomainData
    {
        public required int DatasetId { get; init; }
        public required DatasetInfo? Info { get; init; }
        public required BinaryTokenReader Reader { get; init; }
        public double Weight { get; set; }
    }
}

public enum MixingStrategy
{
    /// <summary>Sample from each domain proportional to its size.</summary>
    Proportional,
    /// <summary>Cycle through domains one batch at a time.</summary>
    RoundRobin,
    /// <summary>Train on a single domain at a time (call NextBatchFromDomain).</summary>
    SingleDomain,
}

public sealed class TrainingBatch : IDisposable
{
    public Tensor InputIds { get; }
    public Tensor TargetIds { get; }
    public int DomainId { get; }

    public TrainingBatch(Tensor inputIds, Tensor targetIds, int domainId)
    {
        InputIds = inputIds;
        TargetIds = targetIds;
        DomainId = domainId;
    }

    public void Dispose()
    {
        InputIds.Dispose();
        TargetIds.Dispose();
    }
}
