function approx_RiemannSolver_kernel!(int_fluid, h, wse, z_b, vx, vy, itype, MAX_NB_FLUID, fluid_neighbor, fluid_dxij, fluid_dyij, fluid_Rij, wall_normals_X, wall_normals_Y, Flux_LeftState_h, Flux_LeftState_hvx, Flux_LeftState_hvy, Flux_RightState_h, Flux_RightState_hvx, 
                                  Flux_RightState_hvy, Flux_Interface_h, Flux_Interface_hvx, Flux_Interface_hvy, BedSource_Left_h, BedSource_Left_hvx, BedSource_Left_hvy, BedSource_Right_h, BedSource_Right_hvx, BedSource_Right_hvy, grav)
    
    tid = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if tid > length(int_fluid)
        return nothing
    end

    i = int_fluid[tid]

    @inbounds for k = 1:MAX_NB_FLUID

        j = fluid_neighbor[k, i]
        j == 0 && continue

        vxi = vx[i]; vyi = vy[i]
        hi  = h[i]; z_bi = z_b[i]; wsei = wse[i]
        hvxi = hi * vxi; hvyi = hi * vyi

       # ---------------- WALL BOUNDARY ----------------
        if itype[j] == -1

            vx_n = (vxi * wall_normals_X[j] + vyi * wall_normals_Y[j]) * wall_normals_X[j]
            vy_n = (vxi * wall_normals_X[j] + vyi * wall_normals_Y[j]) * wall_normals_Y[j]

            vx_t = vxi - vx_n
            vy_t = vyi - vy_n

            vxj = -vx_n + vx_t
            vyj = -vy_n + vy_t

            hj   = hi
            z_bj = z_bi
            wsej = wsei

            hvxj = hj * vxj
            hvyj = hj * vyj

        else
            vxj = vx[j]; vyj = vy[j]
            hj  = h[j]; z_bj = z_b[j]; wsej = wse[j]
            hvxj = hj * vxj; hvyj = hj * vyj
        end

        # ---------------- NORMALS ----------------
        invr = 1.0 / fluid_Rij[k, i]
        nfx  = - fluid_dxij[k, i] * invr
        nfy  = - fluid_dyij[k, i] * invr

        uhL =  vxi * nfx + vyi * nfy
        vhL = -vxi * nfy + vyi * nfx
        uhR =  vxj * nfx + vyj * nfy
        vhR = -vxj * nfy + vyj * nfx

       # ---------------- TORO AVERAGES ----------------
        cL = sqrt(grav * hi)
        cR = sqrt(grav * hj)

        hs  = (0.5*(cL + cR) + 0.25*(uhL - uhR))^2 / grav
        uhs = 0.5*(uhL + uhR) + cL - cR

        sL = hi == 0 ? uhR - 2cR : min(uhL - cL, uhs - sqrt(grav * hs))
        sR = hj == 0 ? uhL + 2cL : max(uhR + cR, uhs + sqrt(grav * hs))

        # ---------------- PRESSURE ----------------
        P_L = 0.5 * grav * hi * hi
        P_R = 0.5 * grav * hj * hj

        # ---------------- BED SOURCE ----------------
        ΔL = max(0.0, z_bi - wsej)
        ΔR = max(0.0, z_bj - wsei)

        Sourceterm = 0.5 * grav * ((hi + hj)*(z_bj - z_bi) + (ΔL*hj - ΔR*hi))

        inv_denom = 1.0 / (sR - sL)

        BedSource_Left_h[k, i] = 0.0
        BedSource_Left_hvx[k, i] = -(sL*inv_denom)*Sourceterm*nfx
        BedSource_Left_hvy[k, i] = -(sL*inv_denom)*Sourceterm*nfy

        BedSource_Right_h[k, i] = 0.0
        BedSource_Right_hvx[k, i] = +(sR*inv_denom)*Sourceterm*nfx
        BedSource_Right_hvy[k, i] = +(sR*inv_denom)*Sourceterm*nfy

        h_L = wsei - ΔL
        h_R = wsej - ΔR

        # ---------------- FLUXES ----------------
        Flux_LeftState_h[k, i]   = hvxi * nfx + hvyi * nfy
        Flux_LeftState_hvx[k, i] = (vxi * hvxi + P_L) * nfx + vyi * hvxi * nfy
        Flux_LeftState_hvy[k, i] = vxi * hvyi * nfx + (vyi * hvyi + P_L) * nfy

        Flux_RightState_h[k, i]   = hvxj * nfx + hvyj * nfy
        Flux_RightState_hvx[k, i] = (vxj * hvxj + P_R) * nfx + vyj * hvxj * nfy
        Flux_RightState_hvy[k, i] = vxj * hvyj * nfx + (vyj * hvyj + P_R) * nfy

        # ---------------- HLL SELECTION ----------------
        if sL >= 0.0

            Flux_Interface_h[k,i]   = Flux_LeftState_h[k,i]
            Flux_Interface_hvx[k,i] = Flux_LeftState_hvx[k,i]
            Flux_Interface_hvy[k,i] = Flux_LeftState_hvy[k,i]

        elseif sR <= 0.0

            Flux_Interface_h[k,i]   = Flux_RightState_h[k,i]
            Flux_Interface_hvx[k,i] = Flux_RightState_hvx[k,i]
            Flux_Interface_hvy[k,i] = Flux_RightState_hvy[k,i]

        else

            FhL1 = hi*uhL
            FhL2 = uhL*(hvxi*nfx + hvyi*nfy) + P_L
            FhR1 = hj*uhR
            FhR2 = uhR*(hvxj*nfx + hvyj*nfy) + P_R
            Fs1 = (sR*FhL1 - sL*FhR1 + sL*sR*(h_R-h_L)) * inv_denom
            Fs2 = (sR*FhL2 - sL*FhR2 + sL*sR*((hvxj*nfx+hvyj*nfy)-(hvxi*nfx+hvyi*nfy))) * inv_denom

            Flux_Interface_h[k,i]   = Fs1
            Flux_Interface_hvx[k,i] = Fs2*nfx - 0.5*(vhL+vhR)*Fs1*nfy
            Flux_Interface_hvy[k,i] = Fs2*nfy + 0.5*(vhL+vhR)*Fs1*nfx

        end
        
    end

    return nothing

end