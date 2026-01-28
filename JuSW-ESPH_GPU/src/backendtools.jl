module BackendTools

    using CUDA

    # -------------------------
    # Backend types
    # -------------------------
    abstract type AbstractBackend end
    struct CPUBackend <: AbstractBackend end
    struct SharedBackend <: AbstractBackend end
    struct GPUBackend <: AbstractBackend end

    # -------------------------
    # Backend detection
    # -------------------------
    default_backend() = GPUBackend()

    # -------------------------
    # Array conversion
    # -------------------------
    to_backend(::CPUBackend, A::AbstractArray) = Array(A)
    to_backend(::GPUBackend, A::AbstractArray) = CuArray(A)
    to_backend(::SharedBackend, A::AbstractArray) = SharedArray(A)

    # -------------------------
    # Tuple support (recursive)
    # -------------------------
    function to_backend(b::AbstractBackend, t::Tuple)
        return tuple((to_backend(b, x) for x in t)...)
    end

    export AbstractBackend, CPUBackend, GPUBackend,
       default_backend, to_backend

end