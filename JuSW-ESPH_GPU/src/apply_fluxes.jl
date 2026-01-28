function apply_fluxes_kernel!(int_fluid, h, volume, itype, MAX_NB_FLUID, fluid_neighbor, fluid_dWijdR, Flux_LeftState_h, Flux_LeftState_hvx, Flux_LeftState_hvy, Flux_Interface_h, Flux_Interface_hvx, Flux_Interface_hvy, BedSource_Left_hvx, BedSource_Left_hvy, d_h_dt, d_hvx_dt, d_hvy_dt)
    
    tid = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if tid > length(int_fluid)
        return
    end

    i = int_fluid[tid]

    hi = h[i]
    local_d_h_dt    = 0.0
    local_d_hvx_dt = 0.0
    local_d_hvy_dt = 0.0

    @inbounds for k = 1:MAX_NB_FLUID

        j = fluid_neighbor[k, i]
        j == 0 && continue

        # skip dry–dry interaction
        hj = h[j]
        if hi == 0.0 && hj == 0.0
            continue
        end
        
        dwdr = fluid_dWijdR[k, i]
        coeff = 2.0 * volume[j] * dwdr

        # ---------------- FLUX DIFFERENCE ----------------
        F1 = Flux_Interface_h[k, i]   - Flux_LeftState_h[k, i]
        F2 = Flux_Interface_hvx[k, i] - Flux_LeftState_hvx[k, i]
        F3 = Flux_Interface_hvy[k, i] - Flux_LeftState_hvy[k, i]

        local_d_h_dt    += coeff * F1
        local_d_hvx_dt += coeff * F2
        local_d_hvy_dt += coeff * F3
        
        # ---------------- FLUID NEIGHBOR ----------------
        if itype[j] == 1
            local_d_hvx_dt += coeff * BedSource_Left_hvx[k, i]
            local_d_hvy_dt += coeff * BedSource_Left_hvy[k, i]
        end

    end

    d_h_dt[i]    = local_d_h_dt
    d_hvx_dt[i]  = local_d_hvx_dt
    d_hvy_dt[i]  = local_d_hvy_dt
        
    return nothing

end
