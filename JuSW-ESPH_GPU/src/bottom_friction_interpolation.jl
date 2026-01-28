function bottom_friction_interpolation_kernel!(int_fluid, h, hvx, hvy, wse, vx, vy, z_b, fr_manning, volume, MAX_NB_BOTTOM, bottom_neighbor, bottom_Wij)
    
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    if tid <= length(int_fluid)

        i = int_fluid[tid]

        local_z_b        = 0.0
        local_fr_manning = 0.0
        local_sum_wij    = 0.0

        @inbounds for k in 1:MAX_NB_BOTTOM

            j = bottom_neighbor[k, i]
            j == 0 && continue

            w = bottom_Wij[k, i]
            
            coeff = volume[j] * w

            local_z_b        += coeff * z_b[j]
            local_fr_manning += coeff * fr_manning[j]
            local_sum_wij    += coeff

        end

        if local_sum_wij > 0.0                    
            
            z_b[i]        = local_z_b / local_sum_wij
            fr_manning[i] = local_fr_manning / local_sum_wij

            h[i]   = max(h[i], wse[i] - z_b[i])
            wse[i] = h[i] + z_b[i]
            hvx[i] = h[i] * vx[i]
            hvy[i] = h[i] * vy[i]

        else

            vx[i] = 0.
            vy[i] = 0.
            h[i]  = 0.
            wse[i] = h[i] + z_b[i]
            hvx[i] = 0.
            hvy[i] = 0.
            
        end
        
    end       
    
    return nothing

end