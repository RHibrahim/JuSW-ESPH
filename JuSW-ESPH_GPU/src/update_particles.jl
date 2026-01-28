function update_particles_half_step_kernel!(int_fluid, h, wse, z_b, vx, vy, hvx, hvy, d_h_dt, d_hvx_dt, d_hvy_dt, dt)

    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    
    if tid <= length(int_fluid)
        
        i = int_fluid[tid]
        
        # --- update water depth ---
        h[i]  = h[i] + (dt/2) * d_h_dt[i]

        # --- update momentum & velocity ---
        hvx[i] = hvx[i] + (dt * 0.5) * d_hvx_dt[i]
        hvy[i] = hvy[i] + (dt * 0.5) * d_hvy_dt[i]

        if h[i] <= 1E-10
            num = h[i] * sqrt(2.0)
            den = sqrt(h[i]^4 + max(h[i]^4, 1E-10))
            vx[i] = hvx[i] * num / den
            vy[i] = hvy[i] * num / den
        else
            vx[i] = hvx[i] / h[i]
            vy[i] = hvy[i] / h[i]
        end

        # --- positivity ---
        if h[i] < 0.0
            h[i] = 0.0
        end

        # --- free surface water ---
        wse[i] = h[i] + z_b[i]

    end

    return nothing
end

function update_particles_full_step_kernel!(int_fluid, h, h_o, wse, z_b, vx, vy, hvx, hvy, hvx_o, hvy_o, d_h_dt, d_hvx_dt, d_hvy_dt, dt)
    
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    if tid <= length(int_fluid)
        
        i = int_fluid[tid]
        
        # --- update water depth ---
        h[i] = h_o[i] + dt * d_h_dt[i]
        
        # --- update momentum & velocity ---            
        hvx[i] = hvx_o[i] + dt * d_hvx_dt[i]
        hvy[i] = hvy_o[i] + dt * d_hvy_dt[i]

        if h[i] <= 1E-10
            num = h[i] * sqrt(2.0)
            den = sqrt(h[i]^4 + max(h[i]^4, 1E-10))
            vx[i] = hvx[i]* num / den
            vy[i] = hvy[i]* num / den
        else
            vx[i] = hvx[i] / h[i]
            vy[i] = hvy[i] / h[i]
        end
        
        
        # --- positivity ---
        if h[i] < 0.0
            h[i] = 0.0
        end

        # --- free surface water---
        wse[i] = h[i] + z_b[i]

    end

    return nothing
end