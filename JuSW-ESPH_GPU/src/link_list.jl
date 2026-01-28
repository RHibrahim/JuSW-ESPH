function link_list!(Gridparams::GridParams, x, y, hsml, itype, MAX_NB_FLUID, MAX_NB_BOTTOM, int_openboundary, int_virtual, ntotal, wall_normals_X, wall_normals_Y, boundary_normals_X, boundary_normals_Y, fluid_neighbor, fluid_Wij, fluid_dWijdR, fluid_dWijdX, fluid_dWijdY, fluid_dxij, fluid_dyij, fluid_Rij, Openboundary_neighbor, Openboundary_Wij, Openboundary_dWijdR, Openboundary_dWijdX, Openboundary_dWijdY, Openboundary_dxij, Openboundary_dyij, Openboundary_Rij, bottom_neighbor, bottom_Wij, bottom_dWijdR, bottom_dWijdX, bottom_dWijdY, bottom_dxij, bottom_dyij, bottom_Rij)

    ngridx     = Gridparams.ngridx
    ngridy     = Gridparams.ngridy
    hsmlcellx  = Gridparams.hsmlcellx
    hsmlcelly  = Gridparams.hsmlcelly
    cellx_id   = Gridparams.cellx_id
    celly_id   = Gridparams.celly_id
    next_cell  = Gridparams.next_cell
    grid_head  = Gridparams.grid_head

    scale_k = (skf == 1) ? 2 : 3

    # --------------------------------------------------
    # Neighbor search
    # --------------------------------------------------
    @inbounds @threads for i in 1:ntotal

        itype[i] ∈ (1,2,-2) || continue
        
        # Initialize interaction counters
        interaction_fluid  = 0
        interaction_bottom = 0
        interaction_ob     = 0
        
        cx = cellx_id[i]
        cy = celly_id[i]

        for gy in max(1,cy-hsmlcelly):min(ngridy,cy+hsmlcelly)
            
            for gx in max(1,cx-hsmlcellx):min(ngridx,cx+hsmlcellx)

                j = grid_head[gx,gy]

                while j != 0
                    
                    if j != i
                        
                        dx = x[i] - x[j]
                        dy = y[i] - y[j]
                        r2  = dx*dx + dy*dy
                        hsmlij = 0.5*(hsml[i] + hsml[j])
                        tol = 1e-6

                        if sqrt(r2) < (scale_k * hsmlij) - tol
 
                            Rij = sqrt(r2)
                                                        
                            # ---------------- fluid → fluid / wall / open boundary interactions
                            if itype[i] == 1 && itype[j] in (1,-1,2,-2)
                                
                                count = interaction_fluid + 1
                                count <= MAX_NB_FLUID || continue

                                interaction_fluid = count

                                fluid_neighbor[count,i] = j
                                fluid_dxij[count,i] = dx
                                fluid_dyij[count,i] = dy
                                fluid_Rij[count,i]    = Rij

                                kernel!(i, Rij, fluid_dxij[count,i], fluid_dyij[count,i], hsmlij, count, fluid_Wij, fluid_dWijdX, fluid_dWijdY, fluid_dWijdR)

                                if itype[j] == -1
                                    wall_normals_X[j] -= fluid_dWijdR[count,i] * (fluid_dxij[count,i] / Rij)
                                    wall_normals_Y[j] -= fluid_dWijdR[count,i] * (fluid_dyij[count,i] / Rij)
                                end

                            end

                            # ---------------- bottom → fluid interactions
                            if itype[i] == 1 && itype[j] == 4
                                
                                count_b = interaction_bottom + 1
                                count_b <= MAX_NB_BOTTOM || continue

                                interaction_bottom = count_b

                                bottom_neighbor[count_b,i] = j
                                bottom_dxij[count_b,i] = dx
                                bottom_dyij[count_b,i] = dy
                                bottom_Rij[count_b,i]  = Rij

                                kernel!(i, Rij, bottom_dxij[count_b,i], bottom_dyij[count_b,i], hsmlij, count_b, bottom_Wij, bottom_dWijdX, bottom_dWijdY, bottom_dWijdR)
                            
                            end

                            # ---------------- open boundary → fluid interactions
                            if itype[i] in (2,-2) && itype[j] == 1
                                
                                count_ob = interaction_ob + 1
                                count_ob <= MAX_NB_FLUID || continue

                                interaction_ob = count_ob

                                Openboundary_neighbor[count_ob,i] = j
                                Openboundary_dxij[count_ob,i] = dx
                                Openboundary_dyij[count_ob,i] = dy
                                Openboundary_Rij[count_ob,i]  = Rij

                                kernel!(i, Rij, Openboundary_dxij[count_ob,i], Openboundary_dyij[count_ob,i], hsmlij, count_ob, Openboundary_Wij, Openboundary_dWijdX, Openboundary_dWijdY, Openboundary_dWijdR)

                                boundary_normals_X[i] += Openboundary_dWijdR[count_ob,i] * (Openboundary_dxij[count_ob,i] / Rij)
                                boundary_normals_Y[i] += Openboundary_dWijdR[count_ob,i] * (Openboundary_dyij[count_ob,i] / Rij)

                            end

                        end

                    end
                    
                    j = next_cell[j]

                end
            end
        end
    end

    # Normalize each normal vector
    @inbounds @simd for i in int_openboundary
        nrm = sqrt(boundary_normals_X[i]^2 + boundary_normals_Y[i]^2)
        if nrm > eps()
            boundary_normals_X[i] /= nrm
            boundary_normals_Y[i] /= nrm
        end
    end

    @inbounds @simd for i in int_virtual
        nrm = sqrt(wall_normals_X[i]^2 + wall_normals_Y[i]^2)
        if nrm > eps()
            wall_normals_X[i] /= nrm
            wall_normals_Y[i] /= nrm
        end
    end

end