using System.IO.MemoryMappedFiles;

namespace TorchSharpLLM.Data;

/// <summary>
/// Memory-mapped token file for zero-copy access to large token datasets.
/// Placeholder for future use with very large datasets that don't fit in RAM.
/// </summary>
public sealed class MemoryMappedTokenFile : IDisposable
{
    private readonly MemoryMappedFile _mmf;
    private readonly MemoryMappedViewAccessor _accessor;

    public int VocabSize { get; }
    public int NumTokens { get; }

    private const int HeaderSize = 8; // 4 bytes vocab + 4 bytes count

    public MemoryMappedTokenFile(string filePath)
    {
        _mmf = MemoryMappedFile.CreateFromFile(filePath, FileMode.Open);
        _accessor = _mmf.CreateViewAccessor();

        VocabSize = _accessor.ReadInt32(0);
        NumTokens = _accessor.ReadInt32(4);
    }

    /// <summary>
    /// Read a single token at the given index.
    /// </summary>
    public int ReadToken(int index) =>
        _accessor.ReadUInt16(HeaderSize + index * 2);

    /// <summary>
    /// Read a sequence of tokens into a pre-allocated array.
    /// </summary>
    public void ReadTokens(int startIndex, long[] buffer, int count)
    {
        long offset = HeaderSize + (long)startIndex * 2;
        for (int i = 0; i < count; i++)
        {
            buffer[i] = _accessor.ReadUInt16(offset);
            offset += 2;
        }
    }

    public void Dispose()
    {
        _accessor.Dispose();
        _mmf.Dispose();
    }
}
