using System.Buffers.Binary;

namespace TorchSharpLLM.Data;

/// <summary>
/// Reads a binary token file produced by tokenize_datasets.py.
///
/// Format:
///   [4 bytes uint32 LE] vocab_size
///   [4 bytes uint32 LE] num_tokens
///   [num_tokens × 2 bytes uint16 LE] token IDs
/// </summary>
public sealed class BinaryTokenReader : IDisposable
{
    private readonly MemoryMappedTokenFile? _mmap;
    private readonly long[] _tokens;

    public int VocabSize { get; }
    public int NumTokens => _tokens.Length;
    public string FilePath { get; }

    private BinaryTokenReader(string filePath, int vocabSize, long[] tokens)
    {
        FilePath = filePath;
        VocabSize = vocabSize;
        _tokens = tokens;
    }

    /// <summary>
    /// Load the entire token file into memory. Fine for files up to a few GB.
    /// </summary>
    public static BinaryTokenReader Load(string filePath)
    {
        if (!File.Exists(filePath))
            throw new FileNotFoundException($"Token file not found: {filePath}");

        using var stream = File.OpenRead(filePath);
        using var reader = new BinaryReader(stream);

        int vocabSize = (int)reader.ReadUInt32();
        int numTokens = (int)reader.ReadUInt32();

        var tokens = new long[numTokens]; // long for TorchSharp tensor compat
        for (int i = 0; i < numTokens; i++)
            tokens[i] = reader.ReadUInt16();

        return new BinaryTokenReader(filePath, vocabSize, tokens);
    }

    /// <summary>
    /// Get a span of tokens starting at the given position.
    /// </summary>
    public ReadOnlySpan<long> GetSpan(int start, int length)
    {
        int end = Math.Min(start + length, _tokens.Length);
        return _tokens.AsSpan(start, end - start);
    }

    /// <summary>
    /// Get tokens as an array slice (for tensor creation).
    /// </summary>
    public long[] GetSlice(int start, int length)
    {
        int actualLength = Math.Min(length, _tokens.Length - start);
        if (actualLength <= 0) return Array.Empty<long>();

        var result = new long[actualLength];
        Array.Copy(_tokens, start, result, 0, actualLength);
        return result;
    }

    /// <summary>
    /// Total number of complete sequences of the given length.
    /// </summary>
    public int CountSequences(int seqLength) =>
        Math.Max(0, NumTokens - seqLength);

    public void Dispose()
    {
        _mmap?.Dispose();
    }
}
