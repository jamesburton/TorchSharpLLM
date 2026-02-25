namespace TorchSharpLLM.Data;

/// <summary>
/// Mirrors the Python dataset catalog — provides metadata about each dataset
/// and its domain-to-expert mapping.
/// </summary>
public static class DatasetCatalog
{
    public static readonly DatasetInfo[] All = new[]
    {
        new DatasetInfo(0, "code_python",      "Code / Programming",       ExpertHint: 0),  // nampdn-ai/tiny-codes
        new DatasetInfo(1, "math_reasoning",   "Mathematics / Reasoning",  ExpertHint: 1),  // meta-math/MetaMathQA
        new DatasetInfo(2, "biomedical",       "Biomedical / Science",     ExpertHint: 2),  // ccdv/pubmed-summarization
        new DatasetInfo(3, "creative_writing", "Creative Writing / Fiction",ExpertHint: 3),  // euclaise/writingprompts
        new DatasetInfo(4, "legal",            "Legal / Regulatory",       ExpertHint: 4),  // pile-of-law/pile-of-law
        new DatasetInfo(5, "finance",          "Finance / Business",       ExpertHint: 5),  // ashraq/financial-news-articles
        new DatasetInfo(6, "conversation",     "Conversational / Dialogue",ExpertHint: 6),  // HuggingFaceH4/ultrachat_200k
        new DatasetInfo(7, "wiki_knowledge",   "General Knowledge",        ExpertHint: 7),  // wikimedia/wikipedia (simple)
        new DatasetInfo(8, "instructions",     "Instruction Following",    ExpertHint: -1), // databricks/databricks-dolly-15k
        new DatasetInfo(9, "textbooks",        "Textbooks / Educational",  ExpertHint: -1), // HuggingFaceTB/cosmopedia-100k
    };

    public static DatasetInfo? ById(int id) =>
        id >= 0 && id < All.Length ? All[id] : null;

    public static DatasetInfo? ByName(string name) =>
        All.FirstOrDefault(d => d.Name == name);
}

public sealed record DatasetInfo(
    int Id,
    string Name,
    string Domain,
    int ExpertHint
);
