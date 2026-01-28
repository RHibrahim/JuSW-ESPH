using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Revise
using Base.Threads
using Printf
using Plots
using DelimitedFiles
using ColorSchemes
using CUDA
using BenchmarkTools
using Glob

include("src/parameters.jl")
include("src/read_data.jl")
include("src/preallocate_arrays.jl")
include("src/JuSW-ESPH.jl")
using .JuSW_ESPH
include("src/visualize_results.jl")

include("src/main.jl")