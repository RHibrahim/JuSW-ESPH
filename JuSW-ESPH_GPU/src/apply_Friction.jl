function apply_Friction_kernel!(int_fluid, vx, vy, h, hvx, hvy, fr_manning, d_hvx_dt, d_hvy_dt, grav)
    
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    if tid <= length(int_fluid)

        i = int_fluid[tid]

        vx_magnitude = sqrt(vx[i]*vx[i] + vy[i]*vy[i])
        h_eff  = max(h[i], 1E-6)
		
		term = fr_manning[i]^2 * vx_magnitude / (h_eff^(4.0/3.0))
		
        d_hvx_dt[i] -= grav * hvx[i] * term
        d_hvy_dt[i] -= grav * hvy[i] * term

    end

    return nothing
    
end