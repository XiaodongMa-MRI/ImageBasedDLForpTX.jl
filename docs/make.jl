using Documenter, ImageBasedDLForpTX

makedocs(;
    modules=[ImageBasedDLForpTX],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.umn.edu/ma000311/ImageBasedDLForpTX.jl/blob/{commit}{path}#L{line}",
    sitename="ImageBasedDLForpTX.jl",
    authors="Xiaodong Ma",
    assets=String[],
)

deploydocs(;
    repo="github.umn.edu/ma000311/ImageBasedDLForpTX.jl",
)
